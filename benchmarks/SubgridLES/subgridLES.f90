module cacheflush_interface
  implicit none

  enum, bind(C) ! lvl_enum_t
      enumerator :: CF_L1_ = 1, &
                    CF_L2_ = 2, &
                    CF_L3_ = 3
  end enum

  interface
    integer(kind=C_int) function cf_init() bind(C,name="cf_init")
      use, intrinsic :: ISO_C_binding, only: C_int
    end function cf_init
    integer(kind=C_int) function cf_flush(lvl_enum) bind(C,name="cf_flush")
      use, intrinsic :: ISO_C_binding, only: C_int
      integer(kind=kind(CF_L1_)), value, intent(in) :: lvl_enum
    end function cf_flush
    integer(kind=C_int) function cf_finalize() bind(C,name="cf_finalize")
      use, intrinsic :: ISO_C_binding, only: C_int
    end function cf_finalize
  end interface
end module

program subgridles_inference
use dalotia_c_interface
use cacheflush_interface
use omp_lib
#ifdef LIKWID_PERFMON
use likwid
#endif ! LIKWID_PERFMON
use,intrinsic::ISO_C_BINDING, only : C_float, C_double
use,intrinsic :: iso_fortran_env, only : int64,real64
  implicit none
    character(len=100) :: filename_model, filename_input, filename_output
    character(len=8), dimension(:), allocatable :: args
    type(C_ptr) :: dalotia_file_pointer

    ! fixed-size input arrays
    real(C_float) :: weight_fc1(10, 300), bias_fc1(300), weight_fc2(300, 6), bias_fc2(6)

    ! increment variables
    integer(kind=int64) :: o, f, i, start_time, end_time, count_rate
    real(kind=real64) :: duration, total_duration = 0
    integer :: num_args, num_inputs_loaded, num_inputs, num_repetitions, num_threads, batch_size, this_thread_num_inputs, thread_num, this_thread_start_index, this_thread_end_index
    integer :: num_input_features = size(weight_fc1, 1)
    integer :: num_hidden_neurons = size(weight_fc1, 2)
    integer :: num_output_features = size(weight_fc2, 2)
    integer(kind=C_int) :: cacheflush_return_value

    ! allocatable input arrays
    real(C_float), dimension(:, :), allocatable :: inputs, temp_inputs, fc1_output, fc2_output, outputs, temp_outputs
    real(C_float), dimension(:, :, :), allocatable :: all_inputs, all_outputs

    num_inputs = 16*16*16
    num_args = command_argument_count()
    allocate(args(num_args))
    if (num_args .gt. 0) then
      call get_command_argument(1, args(1))
      ! parse to num_inputs
      read(args(1), *) num_inputs
    end if

#ifdef DALOTIA_E_FOR_MEMORY_TRACE
    num_repetitions = 1
#else
    num_repetitions = 1000
#endif ! DALOTIA_E_FOR_MEMORY_TRACE
    filename_model = "./weights_SubgridLESNet.safetensors"
    filename_input = "./input_SubgridLESNet.safetensors"
    filename_output = "./output_SubgridLESNet.safetensors"

#ifdef DALOTIA_E_FOR_MEMORY_TRACE
    ! just allocate
    allocate(inputs(num_input_features, num_inputs))
    allocate(outputs(num_output_features, num_inputs))
    allocate(all_inputs(num_input_features, num_inputs, 1))
#else
    write (*, *) "Loading inputs from ", trim(filename_input)
    dalotia_file_pointer = dalotia_open_file(trim(filename_input))
    call dalotia_load_tensor_dense(dalotia_file_pointer, "random_input", inputs)
    call dalotia_close_file(dalotia_file_pointer)
    num_inputs_loaded = size(inputs, 2)
    call assert(num_inputs_loaded == 16*16*16)
    call assert_close_f(inputs(1,1), 0.4963)
    call assert_close_f(inputs(2,1), 0.7682)
    call assert_close_f(inputs(1,2), 0.3489)
    write (*, *) "Loading outputs from ", trim(filename_output)
    dalotia_file_pointer = dalotia_open_file(trim(filename_output))
    call dalotia_load_tensor_dense(dalotia_file_pointer, "output", outputs)
    call dalotia_close_file(dalotia_file_pointer)
    call assert_close_f(outputs(1,1), 2.847215)
    call assert_close_f(outputs(2,1), 0.5240393)
    call assert_close_f(outputs(1,2), 2.555436)
    call assert(size(outputs, 2) == num_inputs_loaded)
    call assert(size(outputs, 1) == num_output_features)

    ! resize inputs/outputs to num_inputs
    write (*, *) "Resizing inputs/outputs to ", num_inputs
    allocate(temp_inputs(num_input_features, num_inputs))
    allocate(temp_outputs(num_output_features, num_inputs))
    do i = 1, num_inputs
      temp_inputs(:, i) = inputs(:, mod(i-1, num_inputs_loaded)+1)
      temp_outputs(:, i) = outputs(:, mod(i-1, num_inputs_loaded)+1)
    end do
    call move_alloc(from=temp_inputs, to=inputs)
    call move_alloc(from=temp_outputs, to=outputs)

    all_inputs = spread(inputs, 3, num_repetitions)
    cacheflush_return_value = cf_init()
    cacheflush_return_value = cf_flush(3)
    if (cacheflush_return_value .ne. 0) then
      error stop "cacheflush failed"
    endif
#endif ! DALOTIA_E_FOR_MEMORY_TRACE
    call assert(size(inputs, 1) == num_input_features)
    ! allocate output array the same size as the read one
    allocate(all_outputs(num_output_features, num_inputs, num_repetitions))

    num_threads = 0
!$OMP parallel reduction(+:num_threads)
    num_threads = num_threads + 1
!$OMP end parallel 

    dalotia_file_pointer = dalotia_open_file(trim(filename_model))
!$OMP parallel default(none) &
!$OMP& shared(all_inputs, num_inputs, num_input_features, num_hidden_neurons, num_repetitions, num_output_features, all_outputs, dalotia_file_pointer, num_threads) &
!$OMP& private (weight_fc1, bias_fc1, weight_fc2, bias_fc2, o, f, i, start_time, end_time, count_rate, duration, fc1_output, fc2_output, this_thread_num_inputs, thread_num, batch_size, this_thread_start_index, this_thread_end_index) &
!$OMP& reduction(max:total_duration)
    ! load model weights
!$OMP critical
    call dalotia_load_tensor(dalotia_file_pointer, "fc1.bias", bias_fc1)
    call dalotia_load_tensor(dalotia_file_pointer, "fc1.weight", weight_fc1)
    call dalotia_load_tensor(dalotia_file_pointer, "fc2.bias", bias_fc2)
    call dalotia_load_tensor(dalotia_file_pointer, "fc2.weight", weight_fc2)
!$OMP end critical

    call assert_close_f(weight_fc1(1,1), 0.3333)
    call assert_close_f(weight_fc1(10,1), 0.0833)
    call assert_close_f(weight_fc1(1,300), 0.00033)
    call assert_close_f(weight_fc1(10,300), 0.00033)
    call assert_close_f(bias_fc1(1), 0.3333)
    call assert_close_f(bias_fc1(300), 0.0033)
    call assert_close_f(weight_fc2(1,1), 1.)
    call assert_close_f(weight_fc2(300,1), 0.00331)
    call assert_close_f(weight_fc2(1,6), 0.00066)
    call assert_close_f(weight_fc2(300,6), 0.000556)
    call assert_close_f(bias_fc2(1), 1.)
    call assert_close_f(bias_fc2(6), 0.16667)

    batch_size = (num_inputs + num_threads - 1) / num_threads;
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
!$OMP single
    write(*,*) "Using ", num_threads, " threads with batch size ", batch_size
!$OMP end single
#endif ! not DALOTIA_E_FOR_MEMORY_TRACE
    allocate(fc1_output(num_hidden_neurons, batch_size))
    allocate(fc2_output(num_output_features, batch_size))
    thread_num = omp_get_thread_num();

!$OMP barrier
    call system_clock(start_time)
#ifdef LIKWID_PERFMON
    call likwid_markerInit()
    call likwid_markerRegisterRegion("SubgridLESNet")
    call likwid_markerStartRegion("SubgridLESNet")
#endif ! LIKWID_PERFMON
    this_thread_start_index = thread_num * batch_size + 1
    this_thread_num_inputs = min(batch_size, num_inputs - thread_num * batch_size);
    if (this_thread_num_inputs .le. 0) then
      error stop "Not enough inputs for the number of threads"
    end if
    this_thread_end_index = this_thread_start_index + this_thread_num_inputs - 1
    do i = 1, num_repetitions
        ! apply fully connected layer
        do o = 1, this_thread_num_inputs
          ! fill with bias
          fc1_output(:,o) = bias_fc1(:)
        end do
        ! this here is more concise but slower due to intermediate arrays: 
        ! fc1_output = spread(bias_fc1, 2, this_thread_num_inputs)
        ! fc1_output = matmul(transpose(weight_fc1), all_inputs(:, this_thread_start_index:this_thread_end_index, i)) + fc1_output

        call sgemm('T', 'N', num_hidden_neurons, this_thread_num_inputs, num_input_features, 1.0, &
                  weight_fc1, num_input_features, all_inputs(:, this_thread_start_index:this_thread_end_index, i), &
                  num_input_features,  1.0, &
                  fc1_output, num_hidden_neurons)

        ! reLU:
        fc1_output = max(0.0, fc1_output)

        do o = 1, this_thread_num_inputs
          fc2_output(:,o) = bias_fc2(:)
        end do

        call sgemm('T', 'N', num_output_features, this_thread_num_inputs, num_hidden_neurons, 1.0, &
                  weight_fc2, num_hidden_neurons, fc1_output, num_hidden_neurons,  1.0, &
                  fc2_output, num_output_features)
        all_outputs(:,this_thread_start_index:this_thread_end_index,i) = fc2_output
!$OMP barrier
    end do
#ifdef LIKWID_PERFMON
    call likwid_markerStopRegion("SubgridLESNet")
    call likwid_markerClose()
#endif ! LIKWID_PERFMON
    call system_clock(end_time)
    call system_clock(count_rate=count_rate)

    duration = real(end_time-start_time, kind=real64)/real(count_rate, kind=real64)

    total_duration = max(total_duration, duration)
!$OMP end parallel

#ifndef DALOTIA_E_FOR_MEMORY_TRACE
    write(*,*) "Duration: ", total_duration, "s"
    write(*,*) "On average: ", total_duration/real(num_repetitions, kind=real64), "s"

  ! compare output
    do i = 1, num_repetitions
      do o = 1, num_inputs
        do f = 1, num_output_features
          call assert_close_f(all_outputs(f, o, i), outputs(f, o))
        end do
      end do
    end do
    cacheflush_return_value = cf_finalize()
#endif ! not DALOTIA_E_FOR_MEMORY_TRACE
  call dalotia_close_file(dalotia_file_pointer)
contains

!cf. https://stackoverflow.com/a/55376595
subroutine raise_exception(message)
  integer i
  character(len=*) message
  print *,message
  i=1
  i=1/(i-i)
  error stop 
end subroutine raise_exception

subroutine assert(condition)
  logical, intent(in) :: condition
  if (.not. condition) then
    call raise_exception("Assertion failed")
  end if
end subroutine assert

subroutine assert_close_f(a, b)
  real, intent(in) :: a, b
  if (abs(a - b) > 1e-4) then
    write (*, *) "Expected ", a, ", got ", b
    call assert(.false.)
  end if
end subroutine assert_close_f

end program subgridles_inference
