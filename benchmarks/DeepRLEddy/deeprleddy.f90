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

program deeprleddy_inference
use dalotia_c_interface
use cacheflush_interface
#ifdef LIKWID_PERFMON
use likwid
#endif ! LIKWID_PERFMON
use,intrinsic::ISO_C_BINDING, only : C_float, C_double
use,intrinsic :: iso_fortran_env, only : int64,real64
  implicit none
    character(len=100) :: filename_input, filename_output
    character(len=8), dimension(:), allocatable :: args
    type(C_ptr) :: dalotia_file_pointer

    ! increment variables
    integer(kind=int64) :: i, o, r
    real(kind=real64) :: duration
    integer :: num_args, num_inputs_loaded, num_inputs, num_repetitions, num_threads, batch_size

    integer :: num_input_features = 3
    integer(kind=C_int) :: cacheflush_return_value

    ! allocatable input arrays
    real(C_float), dimension(:), allocatable :: outputs, temp_outputs
    real(C_float), dimension(:, :), allocatable :: all_outputs
    real(C_float), dimension(:, :, :, :, :), allocatable :: inputs, temp_inputs
    real(C_float), dimension(:, :, :, :, :, :), allocatable :: all_inputs

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
    filename_input = "./input_DeepRLEddyNet.safetensors"
    filename_output = "./output_DeepRLEddyNet.safetensors"

#ifdef DALOTIA_E_FOR_MEMORY_TRACE
    ! just allocate
    allocate(inputs(6, 6, 6, 3, num_inputs))
    allocate(outputs(num_inputs))
    allocate(all_inputs(6, 6, 6, 3, num_inputs, 1))
#else
    write (*, *) "Loading inputs from ", trim(filename_input)
    dalotia_file_pointer = dalotia_open_file(trim(filename_input))
    call dalotia_load_tensor_dense(dalotia_file_pointer, "random_input", inputs)
    call dalotia_close_file(dalotia_file_pointer)
    num_inputs_loaded = size(inputs, 5)
    call assert(num_inputs_loaded == 16*16*16)
    call assert(size(inputs, 4) == 3)
    call assert(size(inputs, 3) == 6)
    call assert(size(inputs, 2) == 6)
    call assert(size(inputs, 1) == 6)
    write (*, *) "Loading outputs from ", trim(filename_output)
    dalotia_file_pointer = dalotia_open_file(trim(filename_output))
    call dalotia_load_tensor_dense(dalotia_file_pointer, "output", outputs)
    call dalotia_close_file(dalotia_file_pointer)
    call assert(size(outputs, 1) == num_inputs_loaded)

    ! resize inputs/outputs to num_inputs
    write (*, *) "Resizing inputs/outputs to ", num_inputs
    allocate(temp_inputs(6, 6, 6, 3, num_inputs))
    allocate(temp_outputs(num_inputs))
    do i = 1, num_inputs
      temp_inputs(:, :, :, :, i) = inputs(:, :, :, :, mod(i-1, num_inputs_loaded)+1)
      temp_outputs(i) = outputs(mod(i-1, num_inputs_loaded)+1)
    end do
    call move_alloc(from=temp_inputs, to=inputs)
    call move_alloc(from=temp_outputs, to=outputs)

    all_inputs = spread(inputs, 6, num_repetitions)
    call assert(size(all_inputs, 1) == size(inputs,1))
    call assert(size(all_inputs, 2) == size(inputs,2))
    call assert(size(all_inputs, 3) == size(inputs,3))
    call assert(size(all_inputs, 4) == size(inputs,4))
    call assert(size(all_inputs, 5) == size(inputs,5))
    cacheflush_return_value = cf_init()
    cacheflush_return_value = cf_flush(3)
    if (cacheflush_return_value .ne. 0) then
      error stop "cacheflush failed"
    endif
#endif ! DALOTIA_E_FOR_MEMORY_TRACE
    call assert(size(inputs, 4) == num_input_features)
    ! allocate output array the same size as the read one
    allocate(all_outputs(num_inputs, num_repetitions))

    num_threads = 0
!$OMP parallel reduction(+:num_threads)
    num_threads = num_threads + 1
!$OMP end parallel 
    batch_size = (num_inputs + num_threads - 1) / num_threads;
    write(*,*) "Using ", num_threads, " threads with batch size ", batch_size

    call inference_direct_convolution(all_inputs, batch_size, all_outputs, duration)
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
    write(*,*) "Duration: ", duration, "s"
    write(*,*) "On average: ", duration/real(num_repetitions, kind=real64), "s"

  ! compare output
    do r = 1, num_repetitions
      do o = 1, num_inputs
        call assert_close_f(all_outputs(o, r), outputs(o))
      end do
    end do
    cacheflush_return_value = cf_finalize()
#endif ! not DALOTIA_E_FOR_MEMORY_TRACE
contains

subroutine inference_direct_convolution(all_inputs, batch_size, all_outputs, duration)
    implicit none
    character(len=100) :: filename_model
    type(C_ptr) :: dalotia_file_pointer
    real(C_float), dimension(:, :, :, :, :, :), intent(in) :: all_inputs
    integer, intent(in) :: batch_size
    real(C_float), dimension(:, :), intent(out) :: all_outputs
    real(kind=real64), intent(out) :: duration
    integer :: start_time, end_time, count_rate
    integer :: num_repetitions, num_inputs, num_batches
    integer :: r, b, o, i, j, k, l, m, n, f, c
    integer :: num_input_features, num_output_features

    ! fixed-size layer arrays
    real(C_float) :: weight_conv1(8, 3, -1:1,-1:1,-1:1), &
                     weight_conv2(8, 8, -1:1,-1:1,-1:1), &
                     weight_conv3(4, 8, -1:1,-1:1,-1:1), &
                     weight_conv4(1, 4,  2, 2, 2), &
                     bias_conv1(8), &
                     bias_conv2(8), &
                     bias_conv3(4), &
                     bias_conv4(1)
    ! intermediate arrays
    real(C_float) :: conv1_input (3, 8, 8, 8, batch_size)
    real(C_float) :: conv1_output(8, 8, 8, 8, batch_size)
    real(C_float) :: conv2_output(8, 4, 4, 4, batch_size)
    real(C_float) :: conv3_output(4, 2, 2, 2, batch_size)
    real(C_float) :: conv4_output(batch_size)

    num_repetitions = size(all_inputs, 6)
    num_inputs = size(all_inputs, 5)
    num_input_features = size(all_inputs, 4)
    num_output_features = 1
    filename_model = "./weights_DeepRLEddyNet.safetensors"

    dalotia_file_pointer = dalotia_open_file(trim(filename_model))
    call dalotia_load_tensor(dalotia_file_pointer, "conv1.bias", bias_conv1)
    call dalotia_load_tensor(dalotia_file_pointer, "conv2.bias", bias_conv2)
    call dalotia_load_tensor(dalotia_file_pointer, "conv3.bias", bias_conv3)
    call dalotia_load_tensor(dalotia_file_pointer, "conv4.bias", bias_conv4)
    call dalotia_load_tensor(dalotia_file_pointer, "conv1.weight", weight_conv1, permutation=[5, 4, 1, 2, 3]) ! load as HWCF
    call dalotia_load_tensor(dalotia_file_pointer, "conv2.weight", weight_conv2, permutation=[5, 4, 1, 2, 3])
    call dalotia_load_tensor(dalotia_file_pointer, "conv3.weight", weight_conv3, permutation=[5, 4, 1, 2, 3])
    call dalotia_load_tensor(dalotia_file_pointer, "conv4.weight", weight_conv4, permutation=[5, 4, 1, 2, 3])

    conv1_input = 0.0 ! for the padding

    call system_clock(start_time)
#ifdef LIKWID_PERFMON
    call likwid_markerInit()
    call likwid_markerRegisterRegion("DeepRLEddyNet")
    call likwid_markerStartRegion("DeepRLEddyNet")
#endif ! LIKWID_PERFMON
    num_batches = ceiling(real(num_inputs) / batch_size)
    do r = 1, num_repetitions
      do b = 0, num_batches-1 !concurrent (b = 1:num_batches)
        ! apply convolution layers
        do o = 1, min(batch_size, num_inputs-batch_size*b)
          ! num_input_channels = size(weight_conv1, 1)
          ! num_output_channels = size(weight_conv1, 2)
          ! gcc$ ivdep
          do i = 1, 6
            do j = 1, 6
              do k = 1, 6
                do c = 1, size(weight_conv1, 2)
                  ! padding: copy input to padded array in NHWDC format
                  conv1_input(c, k+1, j+1, i+1, o) = all_inputs(k, j, i, c, b*batch_size+o, r)
                end do
              end do
            end do
          end do
        end do
        do o = 1, batch_size
          do i = 1, 6
            do j = 1, 6
              do k = 1, 6
                ! fill with bias
                conv1_output(:, k+1, j+1, i+1, o) = bias_conv1
              end do
            end do
          end do
          do l = -1, 1
            do n = -1, 1
              do m = -1, 1
                ! gcc$ vector
                do i = 2, 7
                  do j = 1, 8
                    do k = 1, 8
                      do c = 1, size(weight_conv1, 2)
                        do f = 1, size(weight_conv1, 1)
                          ! apply 3*3*3 stencil
                          conv1_output(f, k, j, i, o) = conv1_output(f, k, j, i, o) + weight_conv1(f,c,m,n,l) * conv1_input(c, k+m, j+n, i+l, o)
                        end do
                      end do
                    end do
                  end do
                end do
              end do
            end do
          end do
        end do
        !reLU
        conv1_output = max(0.0, conv1_output)
        do o = 1, batch_size
          do i = 1, 4
            do j = 1, 4
              do k = 1, 4
                ! fill with bias
                conv2_output(:, k, j, i, o) = bias_conv2
              end do
            end do
          end do
          do l = -1, 1
            do n = -1, 1
              do m = -1, 1
                do i = 1, 4
                  do j = 1, 4
                    do k = 1, 4
                      do c = 1, size(weight_conv2, 2)
                        do f = 1, size(weight_conv2, 1)
                          ! apply 3*3*3 stencil
                          conv2_output(f, k, j, i, o) = conv2_output(f, k, j, i, o) + weight_conv2(f,c, m,n,l) * conv1_output(c, k+m+2, j+n+2, i+l+2, o)
                        end do
                      end do
                    end do
                  end do
                end do
              end do
            end do
          end do
        end do
        !reLU
        conv2_output = max(0.0, conv2_output)
        do o = 1, batch_size
          do i = 1, 2
            do j = 1, 2
              do k = 1, 2
                ! fill with bias
                conv3_output(:, k, j, i, o) = bias_conv3
              end do
            end do
          end do
          do l = -1, 1
            do n = -1, 1
              do m = -1, 1
                do i = 1, 2
                  do j = 1, 2
                    do k = 1, 2
                      do c = 1, size(weight_conv3, 2)
                        do f = 1, size(weight_conv3, 1)
                          ! apply 3*3*3 stencil
                          conv3_output(f, k, j, i, o) = conv3_output(f, k, j, i, o) + weight_conv3(f,c,m,n,l) * conv2_output(c, k+m+1, j+n+1, i+l+1, o)
                        end do
                      end do
                    end do
                  end do
                end do
              end do
            end do
          end do
        end do
        !reLU
        conv3_output = max(0.0, conv3_output)
        ! fill with bias
        conv4_output = bias_conv4(1)
        do o = 1, batch_size
          do l = 1, 2
            do n = 1, 2
              do m = 1, 2
                do c = 1, size(weight_conv4, 2)
                  ! apply 2*2*2 stencil
                  conv4_output(o) = conv4_output(o) + weight_conv4(1,c,m,n,l) * conv3_output(c, m, n, l, o)
                end do
              end do
            end do
          end do
        end do
        ! half-sigmoid
        all_outputs(b*batch_size+1:b*(batch_size+1),r) = 0.5 * 1. / (1. + exp(-conv4_output));
      end do
    end do
#ifdef LIKWID_PERFMON
    call likwid_markerStopRegion("DeepRLEddyNet")
    call likwid_markerClose()
#endif ! LIKWID_PERFMON
    call system_clock(end_time)
    call system_clock(count_rate=count_rate)
  
    duration = real(end_time-start_time, kind=real64)/real(count_rate, kind=real64)
    call dalotia_close_file(dalotia_file_pointer)
end subroutine inference_direct_convolution

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

end program deeprleddy_inference
