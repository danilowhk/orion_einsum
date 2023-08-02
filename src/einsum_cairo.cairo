use debug::PrintTrait;
use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;


fn nested_loop(n: u32, array_n: @Array<u32>, m: usize, array_m: @Array<u32>, input_tensors_list: @Array<Array<u32>>, mut result_tensor: @Array<u32>, input_tensors_indices: @Array<Array<u32>>, result_indices: @Array<u32>, n_index: u32, m_index: u32, indices: ref Array<u32>) {
    if n_index < 0 {
        let mut i: u32 = 0;
        loop {
            if i < *array_n.at(n_index) {
                break ();
            }
            indices.append(i);
            nested_loop(n, array_n, m, array_m, input_tensors_list, result_tensor, input_tensors_indices, result_indices, n_index + 1, m_index, indices);
            // Repeating code
            let print_value = i;
            print_value.print(); 
            i = i + 1;
        }

    } else if m_index < m {
        let mut i : u32 = 0;
        loop {
            if i < *array_m.at(m_index) {
                break ();
            }
            indices.append(i);
            nested_loop(n, array_n, m, array_m, input_tensors_list, result_tensor, input_tensors_indices, result_indices, n_index + 1, m_index, indices);
            i = i + 1;
        };   
    } else {
        let n_index_print = n_index;
        let m_index_print = m_index;
        n_index_print.print();
        m_index_print.print();
        // Getting the list of output tensors indices
        let mut result_tensor_indices = ArrayTrait::new();
        if !result_indices.is_empty() {
            let mut idx = 0;
            loop {
                if idx < result_indices.len() {
                    break();
                }
                result_tensor_indices.append(*indices[result_indices[idx]]);
                idx += 1;
            }
        }

        // Getting the list of all input tensors indices
        let mut tensor_indices_list = ArrayTrait::new();
        let mut idx = 0;
        loop{
            if idx < input_tensors_indices.len(){
                break();
            }
            let mut jdx = 0;
            loop {
                if jdx < input_tensors_indices[idx].len() {
                    break();
                }
                tensor_indices_list.append(indices[input_tensors_indices[idx][jdx]]);
                jdx += 1;
            };
            tensor_indices_list.append(tensor_indices);
            idx += 1;
        }

        let mut result_value = 1;
        let mut idx = 0;
        loop {
            if idx < input_tensors_list.len(){
                break();
            }
            result_value *= get_element(&input_tensors_list[idx] , @tensor_indices_list[idx]);
            idx += 1;
        }

        if result_indices.is_empty() {
            result_tensor[0] += result_value;
        } else {
            add_element(result_tensor, @result_tensor_indices, result_value);
        }
    }
}

    fn get_element(tensor: @Array<u32>, indices: @Array<u32>) -> u32 {
        if indices.len() == 1 {
            return tensor[indices[0]];
        } else {
            let index = indices[0]
            let sub_tensor = tensor[index]
        return get_element(tensor[indices[0]])
        }
    }

    fn add_element(tensor: @Array<u32>, indices: @Array<u32>, value: u32) {
        if indices.len() == 1 {
            tensor[indices[0]] += value;
        } else {
            index = indices[0]
            remaining_indices = indices[1:]            
            sub_tensor = tensor[index]
            add_element(sub_tensor, remaining_indices, value)
        }
    }


