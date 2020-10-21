/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <FOAM2CSR.hpp>
#include <cuda.h>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define CHECK(call)                                              \
    {                                                            \
        cudaError_t e = call;                                    \
        if (e != cudaSuccess)                                    \
        {                                                        \
            printf("Cuda failure: '%s %d %s'",                   \
                __FILE__, __LINE__, cudaGetErrorString(e));      \
        }                                                        \
    }

// Offset the column indices to transform from local to global
__global__ void localToGlobalColIndices(
    const int nnz,
    const int nrows,
    const int nInternalFaces,
    const int diagIndexGlobal,
    const int lowOffGlobal,
    const int uppOffGlobal,
    int *colIndices)
{
    // Offset by global offset
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nnz; i += blockDim.x * gridDim.x)
    {
        int offset;

        // Depending upon the row, different offsets must be applied
        if (i < nrows)
        {
            offset = diagIndexGlobal;
        }
        else if (i < nrows + nInternalFaces)
        {
            offset = uppOffGlobal;
        }
        else
        {
            offset = lowOffGlobal;
        }

        colIndices[i] += offset;
    }
}

// Apply the pre-existing permutation to the values [and columns]
__global__ void applyPermutation(
    const int totalNnz,
    const int *perm,
    const int *colIndicesTmp,
    const double *valuesTmp,
    int *colIndices,
    double *values,
    bool valuesOnly)
{
    // Permute col indices and values
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < totalNnz; i += blockDim.x * gridDim.x)
    {
        int p = perm[i];

        // In the values only case the column indices and row offsets remain fixed so no update
        if (!valuesOnly)
        {
            colIndices[i] = colIndicesTmp[p];
        }

        values[i] = valuesTmp[p];
    }
}

// Flatten the row indices into the row offsets
__global__ void createRowOffsets(
    int nnz,
    int *rowIndices,
    int *rowOffsets)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nnz; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&rowOffsets[rowIndices[i]], 1);
    }
}

// Updates the values based on the previously determined permutation
void FOAM2CSR::updateValues(
    const int nrows,
    const int nInternalFaces,
    const int extNnz,
    const double *diagVal,
    const double *uppVal,
    const double *lowVal,
    const double *extVal)
{
    // Determine the local non-zeros from the internal faces
    int localNnz = nrows + 2 * nInternalFaces;

    // Add external non-zeros (communicated halo entries)
    int totalNnz = localNnz + extNnz;

    // Copy the values in [ diag, upper, lower, (external) ]
    CHECK(cudaMemcpy(valuesTmp, diagVal, sizeof(double) * nrows, cudaMemcpyDefault));
    CHECK(cudaMemcpy(valuesTmp + nrows, uppVal, sizeof(double) * nInternalFaces, cudaMemcpyDefault));
    CHECK(cudaMemcpy(valuesTmp + nrows + nInternalFaces, lowVal, sizeof(double) * nInternalFaces, cudaMemcpyDefault));
    if (extNnz > 0)
    {
        CHECK(cudaMemcpy(valuesTmp + localNnz, extVal, sizeof(double) * extNnz, cudaMemcpyDefault));
    }

    constexpr int nthreads = 128;
    int nblocks = totalNnz / nthreads + 1;
    applyPermutation<<<nblocks, nthreads>>>(totalNnz, ldu2csrPerm, nullptr, valuesTmp, nullptr, values, true);
    CHECK(cudaStreamSynchronize(0));
}

// Perform the conversion between an LDU matrix and a CSR matrix, possibly distributed
void FOAM2CSR::convertLDU2CSR(
    int nrows,
    int nInternalFaces,
    int diagIndexGlobal,
    int lowOffGlobal,
    int uppOffGlobal,
    const int *upperAddr,
    const int *lowerAddr,
    const int extNnz,
    const int *extRow,
    const int *extCol,
    const double *diagVals,
    const double *upperVals,
    const double *lowerVals,
    const double *extVals)
{
    // Determine the local non-zeros from the internal faces
    int localNnz = nrows + 2 * nInternalFaces;

    // Add external non-zeros (communicated halo entries)
    int totalNnz = localNnz + extNnz;

    // Generate unpermuted index list [0, ..., totalNnz-1]
    int *permTmp;
    CHECK(cudaMalloc(&permTmp, sizeof(int) * totalNnz));
    thrust::sequence(thrust::device, permTmp, permTmp + totalNnz, 0);

    // Fill rowIndicesTmp with [0, ..., n-1], lowerAddr, upperAddr, (extAddr)
    int *rowIndicesTmp;
    CHECK(cudaMalloc(&rowIndicesTmp, sizeof(int) * totalNnz));
    CHECK(cudaMemcpy(rowIndicesTmp, permTmp, nrows * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(rowIndicesTmp + nrows, lowerAddr, nInternalFaces * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(rowIndicesTmp + nrows + nInternalFaces, upperAddr, nInternalFaces * sizeof(int), cudaMemcpyDefault));
    if (extNnz > 0)
    {
        CHECK(cudaMemcpy(rowIndicesTmp + localNnz, extRow, extNnz * sizeof(int), cudaMemcpyDefault));
    }

    // Make space for the row indices and stored permutation
    int *rowIndices;
    CHECK(cudaMalloc(&rowIndices, sizeof(int) * totalNnz));
    CHECK(cudaMalloc(&ldu2csrPerm, sizeof(int) * totalNnz));

    // Sort the row indices and store results in the permutation
    void *tempStorage = NULL;
    size_t tempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, rowIndicesTmp, rowIndices, permTmp, ldu2csrPerm, totalNnz);
    CHECK(cudaMalloc(&tempStorage, tempStorageBytes));
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, rowIndicesTmp, rowIndices, permTmp, ldu2csrPerm, totalNnz);
    CHECK(cudaFree(permTmp));
    CHECK(cudaFree(tempStorage));

    // Make space for the row offsets
    CHECK(cudaMalloc(&rowOffsets, sizeof(int) * (nrows + 1)));
    CHECK(cudaMemset(rowOffsets, 0, sizeof(int) * (nrows + 1)));

    // XXX Taking the non zero per row data from the host could be more
    // efficient, experiment with this in the future
    //cudaMemcpy(rowOffsets, nz_per_row, sizeof(int) * nrows, cudaMemcpyDefault);
    //thrust::exclusive_scan(thrust::device, rowOffsets, rowOffsets + nrows + 1, rowOffsets);

    // Convert the row indices into offsets
    constexpr int nthreads = 128;
    int nblocks = totalNnz / nthreads + 1;
    createRowOffsets<<<nblocks, nthreads>>>(totalNnz, rowIndices, rowOffsets);
    thrust::exclusive_scan(thrust::device, rowOffsets, rowOffsets + nrows + 1, rowOffsets);
    CHECK(cudaFree(rowIndices));

    // Fill rowIndicesTmp with diagVals, upperVals, lowerVals, (extVals)
    CHECK(cudaMalloc(&valuesTmp, totalNnz * sizeof(double)));
    CHECK(cudaMemcpy(valuesTmp, diagVals, nrows * sizeof(double), cudaMemcpyDefault));
    CHECK(cudaMemcpy(valuesTmp + nrows, upperVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));
    CHECK(cudaMemcpy(valuesTmp + nrows + nInternalFaces, lowerVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));
    if (extNnz > 0)
    {
        CHECK(cudaMemcpy(valuesTmp + localNnz, extVals, extNnz * sizeof(double), cudaMemcpyDefault));
    }

    // Concat [0, ..., n-1], upperAddr, lowerAddr (note switched) into column indices
    int *colIndicesTmp;
    CHECK(cudaMalloc(&colIndicesTmp, totalNnz * sizeof(int)));
    CHECK(cudaMemcpy(colIndicesTmp, rowIndicesTmp, nrows * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(colIndicesTmp + nrows, rowIndicesTmp + nrows + nInternalFaces, nInternalFaces * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(colIndicesTmp + nrows + nInternalFaces, rowIndicesTmp + nrows, nInternalFaces * sizeof(int), cudaMemcpyDefault));
    if (extNnz > 0)
    {
        CHECK(cudaMemcpy(colIndicesTmp + localNnz, extCol, extNnz * sizeof(int), cudaMemcpyDefault));
    }

    CHECK(cudaFree(rowIndicesTmp));

    // Construct the global column indices
    nblocks = localNnz / nthreads + 1;
    localToGlobalColIndices<<<nblocks, nthreads>>>(localNnz, nrows, nInternalFaces, diagIndexGlobal, lowOffGlobal, uppOffGlobal, colIndicesTmp);

    // Allocate space to store the permuted column indices and values
    CHECK(cudaMalloc(&colIndices, sizeof(int) * totalNnz));
    CHECK(cudaMalloc(&values, sizeof(double) * totalNnz));

    // Swap column indices based on the pre-determined permutation
    nblocks = totalNnz / nthreads + 1;
    applyPermutation<<<nblocks, nthreads>>>(totalNnz, ldu2csrPerm, colIndicesTmp, valuesTmp, colIndices, values, false);
    CHECK(cudaFree(colIndicesTmp));
}

// XXX Should implement an early abandonment of the
// unnecessary data for capacity optimisation
void FOAM2CSR::discardStructure()
{
}

// Deallocate remaining storage
void FOAM2CSR::finalise()
{
    if(rowOffsets != nullptr)
        CHECK(cudaFree(rowOffsets));
    if(colIndices != nullptr)
        CHECK(cudaFree(colIndices));
    if(values != nullptr)
        CHECK(cudaFree(values));
    if(valuesTmp != nullptr)
        CHECK(cudaFree(valuesTmp));
    if(ldu2csrPerm != nullptr)
        CHECK(cudaFree(ldu2csrPerm));
}
