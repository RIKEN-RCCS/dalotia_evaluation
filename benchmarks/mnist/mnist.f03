
program mnist_inference
use dalotia_c_interface
use,intrinsic::ISO_C_BINDING, only : C_float, C_double
use,intrinsic :: iso_fortran_env, only : int64,real64
  implicit none
    integer, parameter :: total_num_images = 10000
    real :: images(28, 28, total_num_images)
    integer :: labels(total_num_images)
    integer :: batch_size = 64
    integer :: num_batches, batch, start, finish, num_correct, num_incorrect
    character(100) :: filename_model, filename_images, filename_labels, tensor_name_conv1, tensor_name_conv2, tensor_name_fc1
    real(C_float), dimension(:,:,:,:), allocatable :: tensor_weight_conv1, tensor_weight_conv2
    real(C_float), dimension(:,:), allocatable :: tensor_weight_fc1
    real(C_float), dimension(:), allocatable :: tensor_bias_conv1, tensor_bias_conv2, tensor_bias_fc1
    integer, dimension(:), allocatable :: tensor_extents
    integer, dimension(total_num_images) :: predictions
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
    integer :: o, k, i, j, f, r
    real(C_float) :: some_value
    integer(kind=int64) :: start_time, end_time, count_rate

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
    tensor_bias_conv1_fixed = tensor_bias_conv1
    tensor_weight_conv1_fixed = tensor_weight_conv1
    tensor_bias_conv2_fixed = tensor_bias_conv2
    tensor_weight_conv2_fixed = tensor_weight_conv2
    tensor_bias_fc1_fixed = tensor_bias_fc1
    tensor_weight_fc1_fixed = tensor_weight_fc1

    images = read_mnist_scaled(trim(filename_images))
    labels = read_mnist_labels(trim(filename_labels))

    ! assert on the first non-0 value
    call assert_close_f(images(7,8,1), 0.3294)

    ! minibatching
    num_batches = ceiling(10000.0 / batch_size)
    call system_clock(start_time)
    do batch = 1, num_batches
        start = (batch - 1) * batch_size + 1
        finish = min(10000, batch * batch_size)

        !zero intermediate arrays?
        fc1_output = 0.0

        ! copy images to a larger array for padding at the edges
        conv1_padded_input(2:29, 2:29, :) = images(:, :, start:(start+batch_size-1))

        ! apply first convolution
        do concurrent (o = 1:64, k = 1:8, i = 2:29, j = 2:29)
          associate (accumulated_value => conv1_output(j-1, i-1, k, o))
            accumulated_value = sum(conv1_padded_input(j-1:j+1, i-1:i+1, o) * tensor_weight_conv1_fixed(:, :, 1, k)) + &
              & tensor_bias_conv1_fixed(k)
            ! relu activation
            if (accumulated_value < 0.0) then
              accumulated_value = 0.0
            end if
          end associate
        end do

        ! apply first pooling
        do concurrent (o = 1:64, k = 1:8, i = 1:14, j = 1:14)
          pool1_padded_output(j+1, i+1, k, o) = maxval(conv1_output(2*j-1:2*j, 2*i-1:2*i, k, o))
        end do

        ! apply second convolution
        do concurrent (o = 1:64, k = 1:16, i = 2:15, j = 2:15)
          associate (accumulated_value => conv2_output(j-1, i-1, k, o))
            accumulated_value = sum(pool1_padded_output(j-1:j+1, i-1:i+1, :, o) * tensor_weight_conv2_fixed(:, :, :, k)) + &
            & tensor_bias_conv2_fixed(k)
            ! relu activation
            if (accumulated_value < 0.0) then
              accumulated_value = 0.0
            end if
          end associate
        end do

        ! apply second pooling
        do concurrent (o = 1:64, k = 1:16, i = 1:7, j = 1:7)
          pool2_output(j, i, k, o) = maxval(conv2_output(2*j-1:2*j, 2*i-1:2*i, k, o))
        end do

        ! ! apply fully connected layer
        do concurrent (o = 1:64)
          ! fill with bias
          fc1_output(:, o) = tensor_bias_fc1_fixed(:)
        end do
        ! sgemm (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        !       'T',       'N',  10,64,784,1.0,     784,    784  1.0,     10
        ! -> A = tensor_weight_fc1_fixed (784*10), B = pool2_output (784*64), 
        !    C = fc1_output(10*64)
        call sgemm('T', 'N', 10, 64, 784, 1.0, tensor_weight_fc1_fixed, 784, &
              pool2_output, 784,  1.0, fc1_output, 10)

        ! make predictions via softmax
        do concurrent (o = 1:64)
          predictions(start + o - 1) = maxloc(fc1_output(:, o), dim=1) - 1
        end do

        ! if (batch == 1) then
        !   call assert_close_f(conv1_output(1, 1, 1, 1), 0.1796)
        !   call assert_close_f(conv1_output(28, 28, 8, 1), 0.6550)
        !   call assert_close_f(pool1_padded_output(2, 2, 1, 1), 0.1796)
        !   call assert_close_f(pool1_padded_output(15, 15, 8, 1), 0.6550)
        !   call assert_close_f(conv2_output(1,1,1,1), 0.4063)
        !   call assert_close_f(conv2_output(14, 14, 2, 1), 0.1789)
        !   call assert_close_f(conv2_output(1, 1, 14, 1), 0.1906)
        !   call assert_close_f(conv2_output(14, 14, 16, 1), 0.0)
        !   call assert_close_f(pool2_output(1, 1, 1, 1), 0.4063)
        !   call assert_close_f(pool2_output(1, 1, 2, 1), 0.04935)
        !   call assert_close_f(pool2_output(7, 7, 2, 1), 0.49008)
        !   call assert_close_f(pool2_output(1, 1, 14, 1), 0.1906)
        !   call assert_close_f(pool2_output(7, 7, 16, 1), 0.0)
        !   call assert_close_f(pool2_output(1, 1, 1, 1), 0.40625)
        !   call assert_close_f(pool2_output(2, 1, 1, 1), 0.0)
        !   call assert_close_f(pool2_output(2, 2, 1, 1), 6.2347)
        !   call assert_close_f(pool2_output(7, 7, 16, 1), 0.0)
        !   call assert_close_f(fc1_output(1, 1), -80.9247)
        !   call assert_close_f(fc1_output(2, 1), -34.6855)
        !   call assert_close_f(fc1_output(3, 1), -31.1533)
        !   call assert_close_f(fc1_output(4, 1), -12.5210)
        !   call assert_close_f(fc1_output(5, 1), -44.1289)
        !   call assert_close_f(fc1_output(6, 1), -40.3522)
        !   call assert_close_f(fc1_output(7, 1), -139.7097)
        !   call assert_close_f(fc1_output(8, 1), 38.1572)
        !   call assert_close_f(fc1_output(9, 1), -47.0220)
        !   call assert_close_f(fc1_output(10, 1), -7.9544)
        ! end if
    end do
    call system_clock(end_time)
    call system_clock(count_rate=count_rate)

  ! compare labels and predictions
    num_correct = 0
    num_incorrect = 0
    do r = 1, 10000
      if (labels(r) == predictions(r)) then
        num_correct = num_correct + 1
      else
        num_incorrect = num_incorrect + 1
      end if
    end do 
    write(*,*) "Got ", num_correct, "/", total_num_images
    write(*,*) "Duration: ", real(end_time-start_time, kind=real64)/real(count_rate, kind=real64), "s"
    call assert_equal_int(num_incorrect, 138)
    call assert_equal_int(num_correct, 9862)


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
