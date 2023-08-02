import numpy as np
import pytest
import torch
#TODO: Format decoder to be Cairo Friendly
def einsum_input_decoder(einsum_formula, tensors):
    # Split the input and output variables
    input_vars, output_vars = einsum_formula.split('->')
    input_vars_list = input_vars.split(',')  # keep commas to split input_vars

    # Combine tensors and input variables into a list of tuples
    combined = list(zip(input_vars_list, tensors))

    # Initialize the dictionaries
    output_dict = {}
    complete_dict = {}
    size_dict = {}

    # Fill the output dictionary
    if output_vars != '':
        for i, var in enumerate(output_vars):
            output_dict[var] = i

    # Fill the complete dictionary with output variables
    for var, i in output_dict.items():
        complete_dict[var] = i

    # Add remaining input variables to the complete dictionary and fill size_dict
    for input_vars, tensor in combined:
        for i, var in enumerate(input_vars):
            if var not in complete_dict:
                complete_dict[var] = len(complete_dict)
            # Only assign size if it hasn't been assigned yet
            if var not in size_dict:
                size_dict[var] = tensor.shape[i]

    # Transform size_dict into an array
    size_array = np.array([size_dict[var] for var in sorted(complete_dict, key=complete_dict.get)])

    # Generate the output tensor with zeros
    if output_vars != '':
        output_tensor_sizes = [size_dict[var] for var in output_vars]
        result_tensor = torch.zeros(*output_tensor_sizes)
    else:
        result_tensor = torch.zeros(1)

    # Generate input_tensors_indices and result_indices
    input_tensors_indices = [[complete_dict[var] for var in vars] for vars in input_vars_list]
    if output_vars != '':
        result_indices = [complete_dict[var] for var in output_vars]
    else:
        result_indices = []

    # Generate output_array and non_repeated_non_output_array
    output_array = [size_dict[var] for var in output_vars] if output_vars != '' else []
    non_repeated_non_output_array = [size_dict[var] for var in complete_dict if var not in output_vars]

    return complete_dict, size_array, result_tensor, input_tensors_indices, result_indices, output_array, non_repeated_non_output_array



# The einsum_input_decoder function remains the same


@pytest.mark.parametrize("test_input", [
    {
        'formula': 'ijk,kl->ijkl',
        'tensors': [torch.randn(3, 4, 4), torch.randn(4, 2)],
        'expected': {
            'complete_dict': {'i': 0, 'j': 1, 'k': 2, 'l': 3},
            'result_tensor': torch.zeros(3, 4, 4, 2),
            'input_tensors_indices': [[0, 1, 2], [2, 3]],
            'result_indices': [0, 1, 2, 3],
            'output_array': [3, 4, 4, 2],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ij,jk->ijk',
        'tensors': [torch.randn(5, 7), torch.randn(7, 3)],
        'expected': {
            'complete_dict': {'i': 0, 'j': 1, 'k': 2},
            'result_tensor': torch.zeros(5, 7, 3),
            'input_tensors_indices': [[0, 1], [1, 2]],
            'result_indices': [0, 1, 2],
            'output_array': [5, 7, 3],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ij,jk->ik',
        'tensors': [torch.randn(3, 4), torch.randn(4, 2)],
        'expected': {
            'complete_dict': {'i': 0, 'k': 1, 'j': 2},
            'result_tensor': torch.zeros(3, 2),
            'input_tensors_indices': [[0, 2], [2, 1]],
            'result_indices': [0, 1],
            'output_array': [3, 2],
            'non_repeated_non_output_array': [4]
        }
    },
    {
        'formula': 'bi,bi->b',
        'tensors': [torch.randn(3, 4), torch.randn(3, 4)],
        'expected': {
            'complete_dict': {'b': 0, 'i': 1},
            'result_tensor': torch.zeros(3),
            'input_tensors_indices': [[0, 1], [0, 1]],
            'result_indices': [0],
            'output_array': [3],
            'non_repeated_non_output_array': [4]
        }
    },
    {
        'formula': 'ii->i',
        'tensors': [torch.randn(3, 3)],
        'expected': {
            'complete_dict': {'i': 0},
            'result_tensor': torch.zeros(3),
            'input_tensors_indices': [[0, 0]],
            'result_indices': [0],
            'output_array': [3],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ii->',
        'tensors': [torch.randn(3, 3)],
        'expected': {
            'complete_dict': {'i': 0},
            'result_tensor': torch.zeros(1),
            'input_tensors_indices': [[0, 0]],
            'result_indices': [],
            'output_array': [],
            'non_repeated_non_output_array': [3]
        }
    },
        {
        'formula': 'ijk,kl->ijkl',
        'tensors': [torch.randn(3, 4, 4), torch.randn(4, 2)],
        'expected': {
            'complete_dict': {'i': 0, 'j': 1, 'k': 2, 'l': 3},
            'result_tensor': torch.zeros(3, 4, 4, 2),
            'input_tensors_indices': [[0, 1, 2], [2, 3]],
            'result_indices': [0, 1, 2, 3],
            'output_array': [3, 4, 4, 2],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ij,jk->ijk',
        'tensors': [torch.randn(5, 7), torch.randn(7, 3)],
        'expected': {
            'complete_dict': {'i': 0, 'j': 1, 'k': 2},
            'result_tensor': torch.zeros(5, 7, 3),
            'input_tensors_indices': [[0, 1], [1, 2]],
            'result_indices': [0, 1, 2],
            'output_array': [5, 7, 3],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ijk,kl->ijkl',
        'tensors': [torch.randn(2, 3, 4), torch.randn(4, 5)],
        'expected': {
            'complete_dict': {'i': 0, 'j': 1, 'k': 2, 'l': 3},
            'result_tensor': torch.zeros(2, 3, 4, 5),
            'input_tensors_indices': [[0, 1, 2], [2, 3]],
            'result_indices': [0, 1, 2, 3],
            'output_array': [2, 3, 4, 5],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ij,jk->ijk',
        'tensors': [torch.randn(2, 3), torch.randn(3, 2)],
        'expected': {
            'complete_dict': {'i': 0, 'j': 1, 'k': 2},
            'result_tensor': torch.zeros(2, 3, 2),
            'input_tensors_indices': [[0, 1], [1, 2]],
            'result_indices': [0, 1, 2],
            'output_array': [2, 3, 2],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'i,i->i',
        'tensors': [torch.randn(5), torch.randn(5)],
        'expected': {
            'complete_dict': {'i': 0},
            'result_tensor': torch.zeros(5),
            'input_tensors_indices': [[0], [0]],
            'result_indices': [0],
            'output_array': [5],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ij->ji',
        'tensors': [torch.randn(3, 4)],
        'expected': {
            'complete_dict': {'j': 0, 'i': 1},
            'result_tensor': torch.zeros(4, 3),
            'input_tensors_indices': [[1, 0]],
            'result_indices': [0, 1],
            'output_array': [4, 3],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ii->i',
        'tensors': [torch.randn(3, 3)],
        'expected': {
            'complete_dict': {'i': 0},
            'result_tensor': torch.zeros(3),
            'input_tensors_indices': [[0, 0]],
            'result_indices': [0],
            'output_array': [3],
            'non_repeated_non_output_array': []
        }
    },
    {
        'formula': 'ij,jk->ik',
        'tensors': [torch.randn(3, 4), torch.randn(4, 2)],
        'expected': {
            'complete_dict': {'i': 0, 'k': 1, 'j': 2},
            'result_tensor': torch.zeros(3, 2),
            'input_tensors_indices': [[0, 2], [2, 1]],
            'result_indices': [0, 1],
            'output_array': [3, 2],
            'non_repeated_non_output_array': [4]
        }
    },
    {
        'formula': 'ijk,jk->ik',
        'tensors': [torch.randn(2, 3, 2), torch.randn(3, 2)],
        'expected': {
            'complete_dict': {'i': 0, 'k': 1, 'j': 2},
            'result_tensor': torch.zeros(2, 2),
            'input_tensors_indices': [[0, 2, 1], [2, 1]],
            'result_indices': [0, 1],
            'output_array': [2, 2],
            'non_repeated_non_output_array': [3]
        }
    }
    
])
def test_einsum_input_decoder(test_input):
    complete_dict, size_array, result_tensor, input_tensors_indices, result_indices, output_array, non_repeated_non_output_array = einsum_input_decoder(test_input['formula'], test_input['tensors'])

    expected = test_input['expected']
    try:
        assert complete_dict == expected['complete_dict'], "Test failed: complete_dict is incorrect"
        assert torch.equal(result_tensor, expected['result_tensor']), "Test failed: result_tensor is incorrect"
        assert input_tensors_indices == expected['input_tensors_indices'], "Test failed: input_tensors_indices is incorrect"
        assert result_indices == expected['result_indices'], "Test failed: result_indices is incorrect"
        assert output_array == expected['output_array'], "Test failed: output_array is incorrect"
        assert non_repeated_non_output_array == expected['non_repeated_non_output_array'], "Test failed: non_repeated_non_output_array is incorrect"
        print(f"Test with formula {test_input['formula']} passed successfully!")
    except AssertionError as e:
        print(str(e))
        raise