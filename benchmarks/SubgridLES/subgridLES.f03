program subgridles_inference
use dalotia_c_interface
use,intrinsic::ISO_C_BINDING, only : C_float, C_double
use,intrinsic :: iso_fortran_env, only : int64,real64
  implicit none
    character(100) :: filename_model, filename_input, filename_output, tensor_name_fc1
    type(C_ptr) :: dalotia_file_pointer

    ! fixed-size input arrays
    real(C_float) :: inputs(10, 4096), weight_fc1(10, 6), bias_fc1(6), outputs(6, 4096)
    real(C_float) :: fc1_output(6, 4096)

    ! increment variables
    integer(kind=int64) :: o, start_time, end_time, count_rate
    integer :: num_inputs = size(inputs, 2)

    filename_model = "./weights_SubgridLESNet.safetensors"
    filename_input = "./input_SubgridLESNet.safetensors"
    filename_output = "./output_SubgridLESNet.safetensors"
  
    tensor_name_fc1 = "fc1"

    write (*, *) "Loading model from ", trim(filename_model)
    dalotia_file_pointer = dalotia_open_file(trim(filename_model))
    call dalotia_load_tensor(dalotia_file_pointer, trim(tensor_name_fc1) //".bias", bias_fc1)
    call dalotia_load_tensor(dalotia_file_pointer, trim(tensor_name_fc1)//".weight", weight_fc1)
    call dalotia_close_file(dalotia_file_pointer)

    call assert_close_f(weight_fc1(1,1), 0.3333)
    call assert_close_f(weight_fc1(10,1), 0.0833)
    call assert_close_f(weight_fc1(1,6), 0.0188)
    call assert_close_f(weight_fc1(10,6), 0.0161)
    call assert_close_f(bias_fc1(1), 0.2746)
    call assert_close_f(bias_fc1(6), -0.0343)

    write (*, *) "Loading inputs from ", trim(filename_input)
    dalotia_file_pointer = dalotia_open_file(trim(filename_input))
    call dalotia_load_tensor(dalotia_file_pointer, "random_input", inputs)
    call dalotia_close_file(dalotia_file_pointer)
    write (*, *) "Loading outputs from ", trim(filename_output)
    dalotia_file_pointer = dalotia_open_file(trim(filename_output))
    call dalotia_load_tensor(dalotia_file_pointer, "output", outputs)
    call dalotia_close_file(dalotia_file_pointer)

    call assert_close_f(inputs(1,1), 0.4963)
    call assert_close_f(inputs(2,1), 0.7682)
    call assert_close_f(inputs(1,2), 0.3489)
    call assert_close_f(outputs(1,1), 1.0331)
    call assert_close_f(outputs(2,1), 0.0446)
    call assert_close_f(outputs(1,2), 0.8264)

    call system_clock(start_time)
       ! apply fully connected layer
        do concurrent (o = 1:4096)
          ! fill with bias
          fc1_output(:, o) = bias_fc1(:)
        end do

        ! fc1_output = matmul(transpose(weight_fc1), inputs) + fc1_output

        call sgemm('T', 'N', 6, num_inputs, 10, 1.0, weight_fc1, 10, &
              inputs, 10,  1.0, fc1_output, 6)

        ! reLU:
        fc1_output = max(0.0, fc1_output)
    call system_clock(end_time)
    call system_clock(count_rate=count_rate)

    write(*,*) "Duration: ", real(end_time-start_time, kind=real64)/real(count_rate, kind=real64), "s"

  ! compare output
    do o = 1, 4096
      call assert_close_f(fc1_output(1, o), outputs(1, o))
      call assert_close_f(fc1_output(2, o), outputs(2, o))
      call assert_close_f(fc1_output(3, o), outputs(3, o))
      call assert_close_f(fc1_output(4, o), outputs(4, o))
      call assert_close_f(fc1_output(5, o), outputs(5, o))
      call assert_close_f(fc1_output(6, o), outputs(6, o))
    end do
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
