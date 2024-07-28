import torch

import cupy
import re

class Stream:
	ptr = torch.cuda.current_stream().cuda_stream
# end

kernel_Sepconv_updateOutput = '''
	extern "C" __global__ void kernel_Sepconv_updateOutput(
		const int n,
		const float* input,
		const float* vertical,
		const float* horizontal,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblOutput = 0.0;

		const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX      = ( intIndex                                                    ) % SIZE_3(output);
		int kernel_size = SIZE_1(vertical);

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
			for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
				dblOutput += VALUE_4(input, intSample, intDepth, kernel_size*intY + intFilterY, kernel_size*intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
			}
		}

		output[intIndex] = dblOutput;
	} }
'''

kernel_Sepconv_updateGradVertical = '''
	extern "C" __global__ void kernel_Sepconv_updateGradVertical(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* horizontal,
		float* gradVertical
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical) / SIZE_1(gradVertical) ) % SIZE_0(gradVertical);
		const int intFilterY  = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical)                        ) % SIZE_1(gradVertical);
		const int intY        = ( intIndex / SIZE_3(gradVertical)                                               ) % SIZE_2(gradVertical);
		const int intX        = ( intIndex                                                                      ) % SIZE_3(gradVertical);
		int kernel_size = SIZE_1(horizontal);
		
		for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1){
			floatOutput += VALUE_4(gradLoss, intSample, 0, intY, intX) * VALUE_4(input, intSample, 0, kernel_size*intY + intFilterY, kernel_size*intX + intFilterX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX) +
				       VALUE_4(gradLoss, intSample, 1, intY, intX) * VALUE_4(input, intSample, 1, kernel_size*intY + intFilterY, kernel_size*intX + intFilterX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX) +
				       VALUE_4(gradLoss, intSample, 2, intY, intX) * VALUE_4(input, intSample, 2, kernel_size*intY + intFilterY, kernel_size*intX + intFilterX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
		}
		
		gradVertical[intIndex] = floatOutput;
	} }

'''

kernel_Sepconv_updateGradHorizontal = '''
	extern "C" __global__ void kernel_Sepconv_updateGradHorizontal(
		const int n,
		const float* gradLoss,
		const float* input,
		const float* vertical,
		float* gradHorizontal
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float floatOutput = 0.0;

		const int intSample   = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal) / SIZE_1(gradHorizontal) ) % SIZE_0(gradHorizontal);
		const int intFilterX  = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal)                          ) % SIZE_1(gradHorizontal);
		const int intY        = ( intIndex / SIZE_3(gradHorizontal)                                                   ) % SIZE_2(gradHorizontal);
		const int intX        = ( intIndex                                                                            ) % SIZE_3(gradHorizontal);
		int kernel_size = SIZE_1(vertical);

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1){
			floatOutput += VALUE_4(gradLoss, intSample, 0, intY, intX) * VALUE_4(input, intSample, 0, kernel_size*intY + intFilterY, kernel_size*intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX) +
				       VALUE_4(gradLoss, intSample, 1, intY, intX) * VALUE_4(input, intSample, 1, kernel_size*intY + intFilterY, kernel_size*intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX) +
				       VALUE_4(gradLoss, intSample, 2, intY, intX) * VALUE_4(input, intSample, 2, kernel_size*intY + intFilterY, kernel_size*intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX);
		}
		
		gradHorizontal[intIndex] = floatOutput;
	} }
'''

kernel_Sepconv_updateGradInput = '''
	extern "C" __global__ void kernel_Sepconv_updateGradInput(
		const int n,
		const float* gradLoss,
		const float* vertical,
		const float* horizontal,
		float* gradInput
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		
		const int intSample   = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput) ) % SIZE_0(gradInput);
		const int intDepth    = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                     ) % SIZE_1(gradInput);
		const int intY        = ( intIndex / SIZE_3(gradInput)                                         ) % SIZE_2(gradInput);
		const int intX        = ( intIndex                                                             ) % SIZE_3(gradInput);
		int kernel_size = SIZE_1(vertical);

		gradInput[intIndex] = VALUE_4(gradLoss, intSample, intDepth, intY / kernel_size, intX / kernel_size) * VALUE_4(vertical, intSample, intY % kernel_size, intY / kernel_size, intX / kernel_size) * VALUE_4(horizontal, intSample, intX % kernel_size, intY / kernel_size, intX / kernel_size);
	} }
'''

def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction]

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class _FunctionSepconv(torch.autograd.Function):
	@staticmethod
	def forward(self, input, vertical, horizontal):
		self.save_for_backward(input, vertical, horizontal)

		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))

		# assert(intInputHeight - intFilterSize == intOutputHeight - 1)
		# assert(intInputWidth - intFilterSize == intOutputWidth - 1)
		assert(intInputHeight == intOutputHeight * intFilterSize)
		assert(intInputWidth  == intOutputWidth * intFilterSize)

		assert(input.is_contiguous() == True)
		assert(vertical.is_contiguous() == True)
		assert(horizontal.is_contiguous() == True)

		output = input.new_zeros([ intSample, intInputDepth, intOutputHeight, intOutputWidth ])

		if input.is_cuda == True:
			n = output.nelement()
			cupy_launch('kernel_Sepconv_updateOutput', cupy_kernel('kernel_Sepconv_updateOutput', {
				'input': input,
				'vertical': vertical,
				'horizontal': horizontal,
				'output': output
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), output.data_ptr() ],
				stream=Stream
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	@staticmethod
	def backward(self, gradOutput):
		input, vertical, horizontal = self.saved_tensors

		intSample = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))

		assert(intInputHeight == intOutputHeight * intFilterSize)
		assert(intInputWidth  == intOutputWidth * intFilterSize)

		assert(gradOutput.is_contiguous() == True)

		gradInput = input.new_zeros([ intSample, intInputDepth, intInputHeight, intInputWidth ]) if self.needs_input_grad[0] == True else None
		gradVertical = input.new_zeros([ intSample, intFilterSize, intOutputHeight, intOutputWidth ]) if self.needs_input_grad[1] == True else None
		gradHorizontal = input.new_zeros([ intSample, intFilterSize, intOutputHeight, intOutputWidth ]) if self.needs_input_grad[2] == True else None
		

		if input.is_cuda == True:
			n = gradVertical.nelement()
			cupy_launch('kernel_Sepconv_updateGradVertical', cupy_kernel('kernel_Sepconv_updateGradVertical', {
				'gradLoss': gradOutput,
				'input': input,
				'horizontal': horizontal,
				'gradVertical': gradVertical
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, gradOutput.data_ptr(), input.data_ptr(), horizontal.data_ptr(), gradVertical.data_ptr() ],
				stream=Stream
			)
			
			cupy_launch('kernel_Sepconv_updateGradHorizontal', cupy_kernel('kernel_Sepconv_updateGradHorizontal', {
				'gradLoss': gradOutput,
				'input': input,
				'vertical': vertical,
				'gradHorizontal': gradHorizontal
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, gradOutput.data_ptr(), input.data_ptr(), vertical.data_ptr(), gradHorizontal.data_ptr() ],
				stream=Stream
			)

			n_i = gradInput.nelement()
			cupy_launch('kernel_Sepconv_updateGradInput', cupy_kernel('kernel_Sepconv_updateGradInput', {
				'gradLoss': gradOutput,
				'vertical': vertical,
				'horizontal': horizontal,
				'gradInput': gradInput
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n_i, gradOutput.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), gradInput.data_ptr() ],
				stream=Stream
			)

		elif input.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradInput, gradVertical, gradHorizontal
	# end
# end

def FunctionSepconv(tensorInput, tensorVertical, tensorHorizontal):
	return _FunctionSepconv.apply(tensorInput, tensorVertical, tensorHorizontal)
# end

class ModuleSepconv(torch.nn.Module):
	def __init__(self):
		super(ModuleSepconv, self).__init__()
	# end

	def forward(self, tensorInput, tensorVertical, tensorHorizontal):
		return _FunctionSepconv.apply(tensorInput, tensorVertical, tensorHorizontal)
	# end
# end
