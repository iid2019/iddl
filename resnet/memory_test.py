import torch
a = torch.zeros(300000000, dtype=torch.int8, device='cuda')
print('CUDA memory allocated: {:.2f}MB'.format(torch.cuda.memory_allocated() / 2**20))
print('CUDA memory cached: {:.2f}MB'.format(torch.cuda.memory_cached() / 2**20))

del a
print('CUDA memory allocated: {:.2f}MB'.format(torch.cuda.memory_allocated() / 2**20))
print('CUDA memory cached: {:.2f}MB'.format(torch.cuda.memory_cached() / 2**20))

torch.cuda.empty_cache()
print('CUDA memory allocated: {:.2f}MB'.format(torch.cuda.memory_allocated() / 2**20))
print('CUDA memory cached: {:.2f}MB'.format(torch.cuda.memory_cached() / 2**20))

input('Press enter key to continue...')
