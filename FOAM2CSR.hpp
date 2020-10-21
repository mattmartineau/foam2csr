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

#pragma once

struct FOAM2CSR
{
    // CSR device data for AmgX matrix
    int *colIndices;
    int *rowOffsets;
    double *values;

    // Temporary storage for the permutation
    double *valuesTmp;

    // Permutation array to convert between LDU and CSR
    // Also possibly encodes sorting of columns
    int *ldu2csrPerm;

    // Perform the conversion between an LDU matrix and a CSR matrix,
    // possibly distributed
    void convertLDU2CSR(
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
        const double *extVals);

    // Updates the CSR matrix values based on LDU matrix values, permuted with
    // the previously determined permutation
    void updateValues(
        const int nrows,
        const int nInternalFaces,
        const int extNnz,
        const double *diagVal,
        const double *uppVal,
        const double *lowVal,
        const double *extVals);

    // Discard elements of the matrix structure
    void discardStructure();

    // Finalise all data
    void finalise();
};
