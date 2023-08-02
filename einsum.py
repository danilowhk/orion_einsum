import torch
import numpy as np
import pytest
from einsum_decoder import einsum_input_decoder


def nested_loop(n, array_n, m, array_m, input_tensors_list, result_tensor, input_tensors_indices, result_indices, n_index=0, m_index=0, indices=[]):
    # Run first through the dimensions of output_tensor then on values that are not in output_tensor
    if n_index < n:

        i = 0
        while i < array_n[n_index]:
            nested_loop(n, array_n, m, array_m, input_tensors_list, result_tensor, input_tensors_indices, result_indices, n_index + 1, m_index, indices + [i])
            i += 1
    elif m_index < m:

        i = 0
        while i < array_m[m_index]:
            nested_loop(n, array_n, m, array_m, input_tensors_list, result_tensor, input_tensors_indices, result_indices, n_index, m_index + 1, indices + [i])
            i += 1
    else:
        # Calculate the indices for result_tensor based on result_indices
        result_tensor_indices = []
        if len(result_indices) != 0:    
            idx = 0
            while idx < len(result_indices):
                result_tensor_indices.append(indices[result_indices[idx]])
                idx += 1

        # Calculate the indices for each tensor
        tensor_indices_list = []
        idx = 0
        while idx < len(input_tensors_indices):
            tensor_indices = []
            jdx = 0
            while jdx < len(input_tensors_indices[idx]):
                tensor_indices.append(indices[input_tensors_indices[idx][jdx]])
                jdx += 1
            tensor_indices_list.append(tensor_indices)
            idx += 1

        # Perform the multiplication and set the result
        result_value = 1.0
        idx = 0
        while idx < len(input_tensors_list):
            result_value *= get_element(input_tensors_list[idx], tensor_indices_list[idx])
            idx += 1
        if len(result_indices) == 0:
            result_tensor[0] += result_value
        else:
            add_element(result_tensor, result_tensor_indices, result_value)
            result_tensor


# We use a recursive function to handle the variable number of dimensions
def get_element(tensor, indices):
    if len(indices) == 1:
        return tensor[indices[0]]
    else:
        return get_element(tensor[indices[0]], indices[1:])    

def add_element(tensor, indices, value):
    if len(indices) == 1:
        tensor[indices[0]] += value
    else:
        add_element(tensor[indices[0]], indices[1:], value)

@pytest.mark.parametrize('einsum_str, dimensions', [
    ('ijk,kl->ijkl', [(3, 4, 4), (4, 2)]),
    ('ij,jk->ijk', [(5, 7), (7, 3)]),
    ('ijk,kl->ijkl', [(2, 3, 4), (4, 5)]),
    ('ij,jk->ijk', [(2, 3), (3, 2)]),
    ('i,i->i', [(5,), (5,)]),
    ('ij->ji', [(3, 4)]),
    ('ii->i', [(3, 3)]),
    ('ij,jk->ik', [(3, 4), (4, 2)]),
    ('i,j->ij', [(4,), (5,)]),
    ('bij,bjk->bik', [(3, 4, 5), (3, 5, 6)]),
    ('bi,bi->b', [(3, 4), (3, 4)]),
    ('ii->i', [(3, 3)]),
    ('ii->', [(2, 2)]),
    ('ijk->ikj', [(2, 3, 4)]),
    ('ij,j->i', [(3, 4), (4,)]),
    ('ij,ij->', [(3, 4), (3, 4)]),
    ('ij,ij->', [(3, 4), (3, 4)]),
    ('kij,kij->ki', [(2, 3, 4), (2, 3, 4)]),
    ('kij->kj', [(4, 3, 4)]),
    ('ij,jk,kl->', [(2, 3), (3, 4), (4, 5)]),
    ('ij,jk,kl->i', [(2, 3), (3, 4), (4, 5)]),
    ('ijk,kjl,lmn->jm', [(2, 3, 4), (4, 3, 5), (5, 6, 7)]),
])

def test_nested_loop(einsum_str, dimensions):
    # Generate the tensors based on the dimensions
    tensors = [torch.randn(*dim) for dim in dimensions]
    inputs_tensors_list = tensors

    complete_dict, size_array, result_tensor, input_tensors_indices, result_indices, output_array, non_repeated_non_output_array = einsum_input_decoder(einsum_str, inputs_tensors_list)

    nested_loop(len(output_array), output_array, len(non_repeated_non_output_array), non_repeated_non_output_array, inputs_tensors_list, result_tensor, input_tensors_indices, result_indices)
    result_einsum = torch.einsum(einsum_str, *tensors)
    print("test started")

    if torch.allclose(result_tensor, result_einsum):
        print(f"Test passed for einsum_str={einsum_str} and dimensions={dimensions}")
    else:
        print(f"Test failed for einsum_str={einsum_str} and dimensions={dimensions}. Expected:\n{result_einsum}\n, but got:\n{result_tensor}")
        assert False