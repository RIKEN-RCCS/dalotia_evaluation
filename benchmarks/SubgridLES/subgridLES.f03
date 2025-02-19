program subgridles_inference
use dalotia_c_interface
use,intrinsic::ISO_C_BINDING, only : C_float, C_double
use,intrinsic :: iso_fortran_env, only : int64,real64
  implicit none
    character(len=100) :: filename_model, filename_input, filename_output, tensor_name_fc1
    character(len=8), dimension(:), allocatable :: args
    type(C_ptr) :: dalotia_file_pointer

    ! fixed-size input arrays
    real(C_float) :: weight_fc1(10, 6), bias_fc1(6)

    ! increment variables
    integer(kind=int64) :: o, f, i, start_time, end_time, count_rate
    real(kind=real64) :: duration
    integer :: num_args, num_inputs_loaded, num_inputs, num_repetitions
    integer :: num_input_features = size(weight_fc1, 1)
    integer :: num_output_features = size(weight_fc1, 2)

    ! allocatable input arrays
    real(C_float), dimension(:, :), allocatable :: inputs, temp_inputs, fc1_output, outputs, temp_outputs

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
    tensor_name_fc1 = "fc1"

    dalotia_file_pointer = dalotia_open_file(trim(filename_model))
    call dalotia_load_tensor(dalotia_file_pointer, trim(tensor_name_fc1) //".bias", bias_fc1)
    call dalotia_load_tensor(dalotia_file_pointer, trim(tensor_name_fc1)//".weight", weight_fc1)
    call dalotia_close_file(dalotia_file_pointer)

    call assert_close_f(weight_fc1(1,1), 0.3333)
    call assert_close_f(weight_fc1(10,1), 0.0833)
    call assert_close_f(weight_fc1(1,6), 0.0188)
    call assert_close_f(weight_fc1(10,6), 0.0161)
    call assert_close_f(bias_fc1(1), 0.3333)
    call assert_close_f(bias_fc1(6), 0.1250)

#ifdef DALOTIA_E_FOR_MEMORY_TRACE
    ! just allocate
    allocate(inputs(num_input_features, num_inputs))
    allocate(outputs(num_output_features, num_inputs))
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
    call assert_close_f(outputs(1,1), 1.0919)
    call assert_close_f(outputs(2,1), 0.5316)
    call assert_close_f(outputs(1,2), 0.8851)
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

#endif ! DALOTIA_E_FOR_MEMORY_TRACE
    call assert(size(inputs, 1) == num_input_features)
    ! allocate output array the same size as the read one
    allocate(fc1_output(num_output_features, num_inputs))

    call system_clock(start_time)
    do i = 1, num_repetitions
       ! apply fully connected layer
        do o = 1, num_inputs !concurrent (o = 1:num_inputs)
          ! fill with bias
          fc1_output(:, o) = bias_fc1(:)
        end do
        ! this here is more concise but slower: 
        ! fc1_output = spread(bias_fc1, 2, num_inputs)
        ! fc1_output = matmul(transpose(weight_fc1), inputs) + fc1_output

        call sgemm('T', 'N', num_output_features, num_inputs, num_input_features, 1.0, &
                  weight_fc1, num_input_features, inputs, num_input_features,  1.0, &
                  fc1_output, num_output_features)

        ! reLU:
        fc1_output = max(0.0, fc1_output)
    end do

    call system_clock(end_time)
    call system_clock(count_rate=count_rate)

    duration = real(end_time-start_time, kind=real64)/real(count_rate, kind=real64)
#ifndef DALOTIA_E_FOR_MEMORY_TRACE
    write(*,*) "Duration: ", duration, "s"
    write(*,*) "On average: ", duration/real(num_repetitions, kind=real64), "s"

  ! compare output
    do o = 1, num_inputs
      do f = 1, num_output_features
        call assert_close_f(fc1_output(f, o), outputs(f, o))
      end do
    end do
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

end program subgridles_inference
