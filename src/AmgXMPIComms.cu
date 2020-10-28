/**
 * \file AmgXMPIComms.cu
 * \brief ***.
 * \author Pi-Yueh Chuang (pychuang@gwu.edu)
 * \author Matt Martineau (mmartineau@nvidia.com)
 * \date 2015-09-01
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 * \copyright Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *            This project is released under MIT License.
 */

// AmgXWrapper
# include "AmgXSolver.H"


/* \implements AmgXSolver::initMPIcomms */
PetscErrorCode AmgXSolver::initMPIcomms(const MPI_Comm &comm)
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    // duplicate the global communicator
    ierr = MPI_Comm_dup(comm, &globalCpuWorld); CHK;
    ierr = MPI_Comm_set_name(globalCpuWorld, "globalCpuWorld"); CHK;

    // get size and rank for global communicator
    ierr = MPI_Comm_size(globalCpuWorld, &globalSize); CHK;
    ierr = MPI_Comm_rank(globalCpuWorld, &myGlobalRank); CHK;


    // Get the communicator for processors on the same node (local world)
    ierr = MPI_Comm_split_type(globalCpuWorld,
            MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCpuWorld); CHK;
    ierr = MPI_Comm_set_name(localCpuWorld, "localCpuWorld"); CHK;

    // get size and rank for local communicator
    ierr = MPI_Comm_size(localCpuWorld, &localSize); CHK;
    ierr = MPI_Comm_rank(localCpuWorld, &myLocalRank); CHK;


    // set up the variable nDevs
    ierr = setDeviceCount(); CHK;


    // set up corresponding ID of the device used by each local process
    ierr = setDeviceIDs(); CHK;
    ierr = MPI_Barrier(globalCpuWorld); CHK;


    // split the global world into a world involved in AmgX and a null world
    ierr = MPI_Comm_split(globalCpuWorld, gpuProc, 0, &gpuWorld); CHK;

    // get size and rank for the communicator corresponding to gpuWorld
    if (gpuWorld != MPI_COMM_NULL)
    {
        ierr = MPI_Comm_set_name(gpuWorld, "gpuWorld"); CHK;
        ierr = MPI_Comm_size(gpuWorld, &gpuWorldSize); CHK;
        ierr = MPI_Comm_rank(gpuWorld, &myGpuWorldRank); CHK;
    }
    else // for those can not communicate with GPU devices
    {
        gpuWorldSize = MPI_UNDEFINED;
        myGpuWorldRank = MPI_UNDEFINED;
    }

    // split local world into worlds corresponding to each CUDA device
    ierr = MPI_Comm_split(localCpuWorld, devID, 0, &devWorld); CHK;
    ierr = MPI_Comm_set_name(devWorld, "devWorld"); CHK;

    // get size and rank for the communicator corresponding to myWorld
    ierr = MPI_Comm_size(devWorld, &devWorldSize); CHK;
    ierr = MPI_Comm_rank(devWorld, &myDevWorldRank); CHK;

    ierr = MPI_Barrier(globalCpuWorld); CHK;

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::setDeviceCount */
PetscErrorCode AmgXSolver::setDeviceCount()
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    // get the number of devices that AmgX solvers can use
    switch (mode)
    {
        case AMGX_mode_dDDI: // for GPU cases, nDevs is the # of local GPUs
        case AMGX_mode_dDFI: // for GPU cases, nDevs is the # of local GPUs
        case AMGX_mode_dFFI: // for GPU cases, nDevs is the # of local GPUs
            // get the number of total cuda devices
            CHECK(cudaGetDeviceCount(&nDevs));
            ierr = PetscPrintf(localCpuWorld, "Number of GPU devices :: %d \n", nDevs); CHK;

            // Check whether there is at least one CUDA device on this node
            if (nDevs == 0) SETERRQ1(MPI_COMM_WORLD, PETSC_ERR_SUP_SYS,
                    "There is no CUDA device on the node %s !\n", nodeName.c_str());
            break;
        case AMGX_mode_hDDI: // for CPU cases, nDevs is the # of local processes
        case AMGX_mode_hDFI: // for CPU cases, nDevs is the # of local processes
        case AMGX_mode_hFFI: // for CPU cases, nDevs is the # of local processes
        default:
            nDevs = localSize;
            break;
    }

    PetscFunctionReturn(0);
}


/* \implements AmgXSolver::setDeviceIDs */
PetscErrorCode AmgXSolver::setDeviceIDs()
{
    PetscFunctionBeginUser;

    PetscErrorCode      ierr;

    // set the ID of device that each local process will use
    if (nDevs == localSize) // # of the devices and local precosses are the same
    {
        devID = myLocalRank;
        gpuProc = 0;
    }
    else if (nDevs > localSize) // there are more devices than processes
    {
        ierr = PetscPrintf(localCpuWorld, "CUDA devices on the node %s "
                "are more than the MPI processes launched. Only %d CUDA "
                "devices will be used.\n", nodeName.c_str(), localSize); CHK;

        devID = myLocalRank;
        gpuProc = 0;
    }
    else // there more processes than devices
    {
        int     nBasic = localSize / nDevs,
                nRemain = localSize % nDevs;

        if (myLocalRank < (nBasic+1)*nRemain)
        {
            devID = myLocalRank / (nBasic + 1);
            if (myLocalRank % (nBasic + 1) == 0)  gpuProc = 0;
        }
        else
        {
            devID = (myLocalRank - (nBasic+1)*nRemain) / nBasic + nRemain;
            if ((myLocalRank - (nBasic+1)*nRemain) % nBasic == 0) gpuProc = 0;
        }
    }

    // Set the device for each rank
    cudaSetDevice(devID);

    PetscFunctionReturn(0);
}

