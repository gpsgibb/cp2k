/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2016  CP2K developers group                         *
 *****************************************************************************/

//! **************************************************************************
//!> \author Hans Pabst (Intel Corp.)
//! **************************************************************************

#if defined(__LIBXSMM) || (defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM))
#include "libxsmm_acc.h"

#if defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM)
# include <libxstream_begin.h>
#elif defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <iostream>
#include <cstdlib>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(CP2K_CONFIG_PREFIX) && (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL))
# include <mkl_service.h>
#endif
#if defined(CP2K_CONFIG_PREFIX) && defined(__TBBMALLOC)
# include <tbb/scalable_allocator.h>
#endif
#if defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM)
# include <libxstream_end.h>
#elif defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


#if defined(__LIBXSMM)
namespace libxsmm_acc_private {
  /** Internal type-agnostic call-forwarding to CP2K/intel stack processing; this is called by dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_[s|d]. */
  template<typename T>
  void process_mm_stack(const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue, const T* a, const T* b, T* c, int* efficient/*Boolean*/)
  {
    int result = LIBXSMM_ACC_ERROR_CONDITION;
    if (0 != descriptor && 0 != params && 0 != myvalue && 0 != a && 0 != b && 0 != c) {
      result = libsmm_acc_process( // TODO: fix const-correctness in libsmm_acc.h
        params, *myvalue, LIBXSMM_ACC_NPARAMS, libxsmm_acc_elem<T,false>::type, const_cast<T*>(a), const_cast<T*>(b), c,
        descriptor->max_m, descriptor->max_n, descriptor->max_k, descriptor->defined_mnk, 0/*stream*/);
      if (efficient) *efficient = 1;
    }
    switch (result) {
      case LIBXSMM_ACC_ERROR_CONDITION: LIBXSMM_ACC_ABORT("incorrect argument(s)"); break;
      default: if (LIBXSMM_ACC_ERROR_NONE != result) LIBXSMM_ACC_ABORT("unknown error");
    }
  }
  const char *const reconf_env = getenv("CP2K_RECONFIGURE");
  const bool reconfigure = ((reconf_env && *reconf_env) ? 0 != atoi(reconf_env) : true);
} // namespace libxsmm_acc_private


LIBXSMM_ACC_EXTERN void xsmm_acc_abort(const char* filename, int line_number, const char* message)
{
  if (filename && *filename) {
    std::cerr << filename << ':' << line_number << " - " << ((message && *message) ? message : "unknown error") << std::endl/*includes flush*/;
  }
  exit(-1);
}


#if defined(CP2K_CONFIG_PREFIX)

#if defined(__GNUC__)
LIBXSMM_ACC_EXTERN LIBXSMM_ACC_ATTRIBUTE(weak)
#else
LIBXSMM_ACC_EXTERN
#endif
void LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(LIBXSMM_ACC_CONCATENATE(__real_, CP2K_CONFIG_PREFIX), dbcsr_get_default_config))(LIBXSMM_ACC_CONFIG_SIGNATURE_DECL);
LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(LIBXSMM_ACC_CONCATENATE(__wrap_, CP2K_CONFIG_PREFIX), dbcsr_get_default_config))(LIBXSMM_ACC_CONFIG_SIGNATURE_DECL)
{
  static bool initialized = false;
  if (!initialized) {
#if defined(__TBBMALLOC)
    const char *const env_hugepages = getenv("CP2K_HUGEPAGES");
    if (0 == env_hugepages || 0 == *env_hugepages || 0 != atoi(env_hugepages)) {
      scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, 1);
    }
#endif
#if defined(_OPENMP) && defined(KMP_VERSION_MAJOR)
    // setting the stacksize applies independent of nested parallelism (LIBXSMM_ACC_OPENMP)
    //kmp_set_stacksize(52428800);
#endif
#if defined(MKL_ENABLE_AVX512)
    mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
#if defined(__LIBXSMM)
    libxsmm_init();
#endif
    // better leave "CP2K_DRIVER" environment variable undocumented
    // variable takes the internal literal/number representing MM driver
    const char *const env_driver = getenv("CP2K_DRIVER");
    if (env_driver && *env_driver) {
      extern int LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(CP2K_CONFIG_PREFIX, dbcsr_cfg));
      LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(CP2K_CONFIG_PREFIX, dbcsr_cfg)) = atoi(env_driver);
    }
    initialized = true;
  }

  LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(LIBXSMM_ACC_CONCATENATE(__real_, CP2K_CONFIG_PREFIX), dbcsr_get_default_config))(LIBXSMM_ACC_CONFIG_SIGNATURE_USE);
  if (libxsmm_acc_private::reconfigure) {
    static const char *const env_stacksize = getenv("CP2K_STACKSIZE");
    if (mm_stack_size && env_stacksize && *env_stacksize) {
      const int value = atoi(env_stacksize);
      if (0 < value) {
        *mm_stack_size = value;
      }
#if defined(LIBXSMM_ACC_STACKSIZE)
      else {
        *mm_stack_size = LIBXSMM_ACC_STACKSIZE;
      }
#endif
    }
    static const char *const env_commtload = getenv("CP2K_COMMTLOAD");
    if (env_commtload && *env_commtload) {
      const int value = atoi(env_commtload);
      if (comm_thread_load) {
        if (0 <= value) {
          if (0 < value) *comm_thread_load = value;
        }
#if defined(LIBXSMM_ACC_COMM_THREAD_LOAD) && (0 < LIBXSMM_ACC_COMM_THREAD_LOAD)
        else {
          *comm_thread_load = LIBXSMM_ACC_COMM_THREAD_LOAD;
        }
#endif
      }
      if (use_comm_thread) {
        if (0 == value) {
          *use_comm_thread = 0;
        }
#if defined(LIBXSMM_ACC_COMM_THREAD_LOAD) && (0 == LIBXSMM_ACC_COMM_THREAD_LOAD)
        else if (0 > value) {
          *use_comm_thread = 0;
        }
#endif
      }
    }
    static const char *const env_multrec = getenv("CP2K_MULTREC");
    if (multrec_limit && env_multrec && *env_multrec) {
      const int value = atoi(env_multrec);
      if (0 < value) {
        *multrec_limit = value;
      }
#if defined(LIBXSMM_ACC_MULTREC_LIMIT)
      else {
        *multrec_limit = LIBXSMM_ACC_MULTREC_LIMIT;
      }
#endif
    }
#if defined(__MPI_VERSION) && (3 <= __MPI_VERSION)
    static const char *const env_rma = getenv("CP2K_RMA");
    if (env_rma && *env_rma) {
      const int value = atoi(env_rma);
      if (0 != value) {
        if (use_mpi_filtering) *use_mpi_filtering = 0;
        if (use_mpi_exp) *use_mpi_exp = 1;
      }
    }
#endif
#if defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM)
# if defined(LIBXSMM_ACC_ACCDRV_POSTERIOR_STREAMS)
    if (accdrv_posterior_streams) {
      *accdrv_posterior_streams = LIBXSMM_ACC_ACCDRV_POSTERIOR_STREAMS;
    }
# endif
# if defined(LIBXSMM_ACC_ACCDRV_POSTERIOR_BUFFERS)
    if (accdrv_posterior_buffers) {
      *accdrv_posterior_buffers = LIBXSMM_ACC_ACCDRV_POSTERIOR_BUFFERS;
    }
# endif
# if defined(LIBXSMM_ACC_ACCDRV_PRIORITY_STREAMS)
    if (accdrv_priority_streams) {
      *accdrv_priority_streams = LIBXSMM_ACC_ACCDRV_PRIORITY_STREAMS;
    }
# endif
# if defined(LIBXSMM_ACC_ACCDRV_PRIORITY_BUFFERS)
    if (accdrv_priority_buffers) {
      *accdrv_priority_buffers = LIBXSMM_ACC_ACCDRV_PRIORITY_BUFFERS;
    }
# endif
# if defined(LIBXSMM_ACC_ACCDRV_MIN_NFLOPS_PERMM)
    if (accdrv_min_flop_process) {
      *accdrv_min_flop_process = LIBXSMM_ACC_ACCDRV_MIN_NFLOPS_PERMM;
    }
# endif
#endif
  }
}


LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_s)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const float* a, const float* b, float* c, int* efficient/*Boolean*/);
LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(xsmm_acc_process_mm_stack_s)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const float* a, const float* b, float* c, int* efficient/*Boolean*/)
{
  if (libxsmm_acc_private::reconfigure) {
    libxsmm_acc_private::process_mm_stack(
      descriptor, params, myvalue, a, b, c, efficient);
  }
  else { /* CP2K/trunk/master code path */
    LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_s)(
      descriptor, params, myvalue, a, b, c, efficient);
  }
}


LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_d)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const double* a, const double* b, double* c, int* efficient/*Boolean*/);
LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(xsmm_acc_process_mm_stack_d)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const double* a, const double* b, double* c, int* efficient/*Boolean*/)
{
  if (libxsmm_acc_private::reconfigure) {
    libxsmm_acc_private::process_mm_stack(
      descriptor, params, myvalue, a, b, c, efficient);
  }
  else { /* CP2K/trunk/master code path */
    LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_d)(
      descriptor, params, myvalue, a, b, c, efficient);
  }
}

#endif // defined(CP2K_CONFIG_PREFIX)

#endif // defined(__LIBXSMM)
#endif // defined(__LIBXSMM) || (defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM))
