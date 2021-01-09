# receptive-field
There are some applications where the receptive field of your convolutional network is an important parameter to control. I found the following ways to compute it:

- You can solve the appropiate equations of each layer by hand.
- You can use an online calculator that asks you to input each layer one by one such as [this one](https://fomoro.com/research/article/receptive-field-calculator).
- You can use an available code that numerically estimates receptive fields in [Pytorch](https://github.com/rogertrullo/Receptive-Field-in-Pytorch) or [Keras](https://github.com/fornaxai/receptivefield). Numerical estimations of receptive fields are based on backpropagation of gradients, and they are an alternative to the analytical receptive field computed by the first two options. Numerical estimations are also great to visualize the *_effective_* receptive field, which could be smaller than the theoretical one.

For my use case, however, the first two options were not scaling well, and the third option was not what I was looking for, so I wrote this simple code in Python to **_mock_** a convolutional architecture and automatically compute its theoretical receptive field. That allowed me to specify the architecture through common blocks and for loops, making the process easier and faster when compared to the online calculator (and of course the manual calculations).

In this code we are mocking the neural network in the sense that each layer is represented by the kernel size, the stride, and the dilation rate. That is, we are not building/using the *_real_* neural network for simplicity. Although this means duplicated code (you are not using your real model if you already built it), it is not an important overhead in practice and the mocking process should feel very familiar to you if you are used to Pytorch or Keras.

The mocking classes are contained in the "mocks.py" module and several examples are presented in the jupyter notebook called "receptive_field_demo".
