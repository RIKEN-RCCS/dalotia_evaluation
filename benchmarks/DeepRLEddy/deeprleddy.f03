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
    character(len=100) :: filename_model, filename_input, filename_output
    character(len=8), dimension(:), allocatable :: args
    type(C_ptr) :: dalotia_file_pointer

    ! fixed-size input arrays
    real(C_float) :: weight_conv1(-1:1,-1:1,-1:1,3,8), weight_conv2(-1:1,-1:1,-1:1,8,8), weight_conv3(-1:1,-1:1,-1:1,8,4), weight_conv4(2, 2, 2, 4, 1), bias_conv1(8), bias_conv2(8), bias_conv3(4), bias_conv4(1)
    ! intermediate arrays
    real(C_float) :: conv1_input(8, 8, 8, 3),conv1_output(8, 8, 8, 8),conv2_output(4, 4, 4, 8),conv3_output(2, 2, 2, 4), conv4_output

    ! increment variables
    integer(kind=int64) :: o, f, i, j, c, k, l, m, n, r, start_time, end_time, count_rate
    real(kind=real64) :: duration
    integer :: num_args, num_inputs_loaded, num_inputs, num_repetitions, num_input_channels, num_output_channels, stencil_size
    integer :: num_input_features = size(weight_conv1, 4)
    integer :: num_output_features = 1000
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
    filename_model = "./weights_DeepRLEddyNet.safetensors"
    filename_input = "./input_DeepRLEddyNet.safetensors"
    filename_output = "./output_DeepRLEddyNet.safetensors"

    dalotia_file_pointer = dalotia_open_file(trim(filename_model))
    call dalotia_load_tensor(dalotia_file_pointer, "conv1.bias", bias_conv1)
    call dalotia_load_tensor(dalotia_file_pointer, "conv2.bias", bias_conv2)
    call dalotia_load_tensor(dalotia_file_pointer, "conv3.bias", bias_conv3)
    call dalotia_load_tensor(dalotia_file_pointer, "conv4.bias", bias_conv4)
    call dalotia_load_tensor(dalotia_file_pointer, "conv1.weight", weight_conv1)
    call dalotia_load_tensor(dalotia_file_pointer, "conv2.weight", weight_conv2)
    call dalotia_load_tensor(dalotia_file_pointer, "conv3.weight", weight_conv3)
    call dalotia_load_tensor(dalotia_file_pointer, "conv4.weight", weight_conv4)
    call dalotia_close_file(dalotia_file_pointer)
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
    conv1_input = 0.0
    cacheflush_return_value = cf_init()
    cacheflush_return_value = cf_flush(3)
    if (cacheflush_return_value .ne. 0) then
      error stop "cacheflush failed"
    endif
#endif ! DALOTIA_E_FOR_MEMORY_TRACE
    call assert(size(inputs, 4) == num_input_features)
    ! allocate output array the same size as the read one
    allocate(all_outputs(num_inputs, num_repetitions))

    call system_clock(start_time)
#ifdef LIKWID_PERFMON
    call likwid_markerInit()
    call likwid_markerRegisterRegion("DeepRLEddyNet")
    call likwid_markerStartRegion("DeepRLEddyNet")
#endif ! LIKWID_PERFMON
    do r = 1, num_repetitions
        ! apply convolution layers
        do o = 1, num_inputs !concurrent (io = 1:num_inputs)
          num_input_channels = size(weight_conv1, 4)
          num_output_channels = size(weight_conv1, 5)
          do c = 1, num_input_channels
            do i = 1, 6
              do j = 1, 6
                do k = 1, 6
                  ! padding: copy input to padded array
                  conv1_input(k+1, j+1, i+1, c) = all_inputs(k, j, i, c, o, r)
                end do
              end do
            end do
          end do
          do i = 2, 7
            do j = 2, 7
              do k = 2, 7
                ! fill with bias
                conv1_output(k, j, i, :) = bias_conv1
              end do
            end do
          end do
          do f = 1, num_output_channels
            do c = 1, num_input_channels
              do l = -1, 1
                do n = -1, 1
                  do m = -1, 1
                    ! dir$ vector
                    ! dir$ ivdep
                    ! dir$ simd
                    do i = 2, 7
                      do j = 1, 8
                        do k = 1, 8
                          ! apply 3*3*3 stencil
                          conv1_output(k, j, i, f) = conv1_output(k, j, i, f) + weight_conv1(m,n,l,c,f) * conv1_input(k+m, j+n, i+l, c)
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
          num_input_channels = size(weight_conv2, 4)
          num_output_channels = size(weight_conv2, 5)
          do i = 1, 4
            do j = 1, 4
              do k = 1, 4
                ! fill with bias
                conv2_output(k, j, i, :) = bias_conv2
              end do
            end do
          end do
          do f = 1, num_output_channels
            do c = 1, num_input_channels
              do l = -1, 1
                do n = -1, 1
                  do m = -1, 1
                    ! dir$ vector
                    ! dir$ ivdep
                    ! dir$ simd
                    do i = 1, 4
                      do j = 1, 4
                          do k = 1, 4
                          ! apply 3*3*3 stencil
                          conv2_output(k, j, i, f) = conv2_output(k, j, i, f) + weight_conv2(m,n,l,c,f) * conv1_output(k+m+2, j+n+2, i+l+2, c)
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
          num_input_channels = size(weight_conv3, 4)
          num_output_channels = size(weight_conv3, 5)
          do i = 1, 2
            do j = 1, 2
              do k = 1, 2
                ! fill with bias
                conv3_output(k, j, i, :) = bias_conv3
              end do
            end do
          end do
          do f = 1, num_output_channels
            do c = 1, num_input_channels
              do l = -1, 1
                do n = -1, 1
                  ! dir$ vector
                  ! dir$ ivdep
                  ! dir$ simd
                  do m = -1, 1
                    do i = 1, 2
                      do j = 1, 2
                        do k = 1, 2
                          ! apply 3*3*3 stencil
                          conv3_output(k, j, i, f) = conv3_output(k, j, i, f) + weight_conv3(m,n,l,c,f) * conv2_output(k+m+1, j+n+1, i+l+1, c)
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
          num_input_channels = size(weight_conv4, 4)
          num_output_channels = size(weight_conv4, 5)
          ! fill with bias
          conv4_output = bias_conv4(1)
          ! dir$ vector
          ! dir$ ivdep
          ! dir$ simd
          do c = 1, num_input_channels
            do l = 1, 2
              do n = 1, 2
                do m = 1, 2
                  ! apply 2*2*2 stencil
                  conv4_output = conv4_output + weight_conv4(m,n,l,c,1) * conv3_output(m, n, l, c)
                end do
              end do
            end do
          end do
          ! half-sigmoid
          all_outputs(o,r) = 0.5 * 1. / (1. + exp(-conv4_output));
        end do
    end do
#ifdef LIKWID_PERFMON
    call likwid_markerStopRegion("DeepRLEddyNet")
    call likwid_markerClose()
#endif ! LIKWID_PERFMON
    call system_clock(end_time)
    call system_clock(count_rate=count_rate)
    write(*,*) conv1_output(2, 2, 2, :)
  

    duration = real(end_time-start_time, kind=real64)/real(count_rate, kind=real64)
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

!cf. https://stackoverflow.com/a/55376595
subroutine raise_exception(message)
  integer i
  character(len=*) message
  print *,message
  i=1
  i=1/(i-i)
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
