program subgridles_inference
use dalotia_c_interface
use,intrinsic::ISO_C_BINDING, only : C_float, C_double
use,intrinsic :: iso_fortran_env, only : int64,real64
  implicit none
    character(100) :: filename_model, filename_input, filename_output, tensor_name_fc1
    type(C_ptr) :: dalotia_file_pointer

    ! fixed-size input arrays
    real(C_float) :: inputs(10, 4096), weight_fc1(10, 6), bias_fc1(6), outputs(6, 4096)

    filename_model = "./weights_SubgridLESNet.safetensors"
    filename_input = "./input_SubgridLESNet.safetensors"
    filename_output = "./output_SubgridLESNet.safetensors"
  
    tensor_name_fc1 = "fc1"

    write (*, *) "Loading model from ", trim(filename_model)
    dalotia_file_pointer = dalotia_open_file(trim(filename_model))
    call dalotia_load_tensor(dalotia_file_pointer, trim(tensor_name_fc1) //".bias", bias_fc1)
    call dalotia_load_tensor(dalotia_file_pointer, trim(tensor_name_fc1)//".weight", weight_fc1)
    call dalotia_close_file(dalotia_file_pointer)
    dalotia_file_pointer = dalotia_open_file(trim(filename_input))
    call dalotia_load_tensor(dalotia_file_pointer, "random_input", inputs)
    call dalotia_close_file(dalotia_file_pointer)
    dalotia_file_pointer = dalotia_open_file(trim(filename_output))
    call dalotia_load_tensor(dalotia_file_pointer, "output", outputs)
    call dalotia_close_file(dalotia_file_pointer)

end program subgridles_inference
