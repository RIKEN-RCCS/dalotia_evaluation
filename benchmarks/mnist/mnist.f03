
program mnist_inference
use dalotia_c_interface
use,intrinsic::ISO_C_BINDING
  implicit none
    real :: images(28, 28, 10000)
    integer :: labels(10000)
    integer :: batch_size = 64
    integer :: num_batches, batch, start, finish
    character(100) :: filename_model, filename_images, filename_labels, tensor_name_conv1, tensor_name_conv2, tensor_name_fc1
    real(C_float), dimension(:,:,:,:), allocatable :: tensor_weight_conv1, tensor_weight_conv2
    real(C_float), dimension(:,:), allocatable :: tensor_weight_fc1
    real(C_float), dimension(:), allocatable :: tensor_bias_conv1, tensor_bias_conv2, tensor_bias_fc1
    integer, dimension(:), allocatable :: tensor_extents
    integer, dimension(10000) :: predictions
    type(C_ptr) :: dalotia_file_pointer

    ! fixed-size model arrays
    real(C_float), dimension(3, 3, 1, 8) :: tensor_weight_conv1_fixed
    real(C_float), dimension(8) :: tensor_bias_conv1_fixed
    real(C_float), dimension(3, 3, 8, 16) :: tensor_weight_conv2_fixed
    real(C_float), dimension(16) :: tensor_bias_conv2_fixed
    real(C_float), dimension(784, 10) :: tensor_weight_fc1_fixed
    real(C_float), dimension(10) :: tensor_bias_fc1_fixed

    ! inference intermediate arrays
    real(C_float), dimension(30, 30, 64) :: conv1_padded_input
    real(C_float), dimension(28, 28, 8, 64) :: conv1_output
    real(C_float), dimension(16, 16, 8, 64) :: pool1_padded_output
    real(C_float), dimension(14, 14, 16, 64) :: conv2_output
    real(C_float), dimension(7, 7, 16, 64) :: pool2_output
    real(C_float), dimension(10, 64) :: fc1_output

    ! increment variables
    integer :: o, k, i, j

    filename_model = "./model-mnist.safetensors"
    filename_images = "./t10k-images-idx3-ubyte"
    filename_labels = "./t10k-labels-idx1-ubyte"

    tensor_name_conv1 = "conv1"
    tensor_name_conv2 = "conv2"
    tensor_name_fc1 = "fc1"

    write (*, *) "Loading model from ", trim(filename_model)
    dalotia_file_pointer = dalotia_open_file(trim(filename_model))
    call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_conv1) //".bias", tensor_bias_conv1)
    call dalotia_get_tensor_extents(dalotia_file_pointer, trim(tensor_name_conv1)//".bias", tensor_extents)
    call assert_equal_int(tensor_extents(1), 8)
    call assert_equal_int(ubound(tensor_bias_conv1, 1), 8)
    call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_conv1)//".weight", tensor_weight_conv1)
    call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_conv2) //".bias", tensor_bias_conv2)
    call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_conv2)//".weight", tensor_weight_conv2)
    call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_fc1) //".bias", tensor_bias_fc1)
    call dalotia_load_tensor_dense(dalotia_file_pointer, trim(tensor_name_fc1)//".weight", tensor_weight_fc1)
    call dalotia_close_file(dalotia_file_pointer)

    !copy to the fixed-size arrays
    tensor_bias_conv1_fixed = tensor_bias_conv1(1:8)
    tensor_weight_conv1_fixed = tensor_weight_conv1(1:3, 1:3, 1:1, 1:8)
    tensor_bias_conv2_fixed = tensor_bias_conv2(1:16)
    tensor_weight_conv2_fixed = tensor_weight_conv2(1:3, 1:3, 1:8, 1:16)
    tensor_bias_fc1_fixed = tensor_bias_fc1(1:10)
    tensor_weight_fc1_fixed = tensor_weight_fc1(1:784, 1:10)

    images = read_mnist_scaled(trim(filename_images))
    labels = read_mnist_labels(trim(filename_labels))

    ! assert on the first non-0 value
    call assert_close_f(images(7,8,1), 0.3294)

    ! minibatching
    num_batches = ceiling(10000.0 / batch_size)
    do batch = 1, num_batches
        start = (batch - 1) * batch_size + 1
        finish = min(10000, batch * batch_size)

        !zero all intermediate arrays
        conv1_output = 0.0
        pool1_padded_output = 0.0
        conv2_output = 0.0
        pool2_output = 0.0
        fc1_output = 0.0

        ! copy images to a larger array for padding at the edges
        conv1_padded_input(2:29, 2:29, :) = images(:, :, start:(start+batch_size-1))

        ! apply convolution
        do o = 1, 64
          do k = 1, 8
            do i = 2, 29
              do j = 2, 29
                conv1_output(j-1, i-1, k, o) = conv1_output(j-1, i-1, k, o) + &
                  & sum(conv1_padded_input(j-1:j+1, i-1:i+1, o) * tensor_weight_conv1_fixed(:, :, 1, k)) + &
                  & tensor_bias_conv1_fixed(k)
              end do
            end do
          end do
        end do


        if (batch == 1) then
          call assert_close_f(conv1_output(1, 1, 1, 1), 0.1796)
          call assert_close_f(conv1_output(28, 28, 8, 1), 0.6550)
        end if
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

subroutine assert_equal_int(a, b)
  integer, intent(in) :: a, b
  if (a /= b) then
    write (*, *) "Expected ", a, ", got ", b
    call assert(.false.)
  end if
end subroutine assert_equal_int

subroutine assert_close_f(a, b)
  real, intent(in) :: a, b
  if (abs(a - b) > 1e-4) then
    write (*, *) "Expected ", a, ", got ", b
    call assert(.false.)
  end if
end subroutine assert_close_f

function read_mnist_scaled(full_path) result (array_of_images)
  use, intrinsic :: iso_fortran_env, only: INT32, INT8
  implicit none

  character(*), intent(in) :: full_path
  real, dimension(28, 28, 10000) :: array_of_images
  
  integer(INT32) :: magic_number
  integer(INT32) :: number_of_images
  integer(INT32) :: n_rows
  integer(INT32) :: n_columns
  integer(INT8)  :: images_int8(28, 28, 10000)
  integer :: file_handle
  
  open (newunit = file_handle, file = full_path, action = 'read', form = 'unformatted', &
  &     access = 'stream', status = 'old', convert = 'big_endian')
  
  read (file_handle) magic_number, number_of_images, n_rows, n_columns, images_int8
  close (file_handle)
  call assert_equal_int(magic_number, 2051)
  
  array_of_images = real(iand(int(images_int8), 255)) / 255.0  ! unsigned 8-bit integer -> default integer -> real
end function read_mnist_scaled

function read_mnist_labels(full_path) result (array_of_labels)
  use, intrinsic :: iso_fortran_env, only: INT32, INT8
  implicit none

  character(*), intent(in) :: full_path
  
  integer(INT32) :: magic_number
  integer(INT32) :: number_of_images
  integer(INT8)  :: array_of_labels(10000)
  integer :: file_handle
  
  open (newunit = file_handle, file = full_path, action = 'read', form = 'unformatted', &
  &     access = 'stream', status = 'old', convert = 'big_endian')
  
  read (file_handle) magic_number, number_of_images, array_of_labels
  close (file_handle)
  call assert_equal_int(magic_number, 2049)
end function read_mnist_labels

end program mnist_inference
