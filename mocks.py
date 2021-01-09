"""Module that contains the mocking objects."""


class MockLayer(object):
    """Generic mocking layer"""
    def __init__(self, kernel_size, stride, dilation_rate, name=''):
        """
        :param kernel_size: (integer) size of the kernel.
        :param stride: (integer) size of the stride.
        :param dilation_rate: (integer) the dilation rate of the kernel.
        :param name: (string) an optional name of the layer (it defaults to '').
        """

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.name = name

    def get_as_list(self):
        """Returns the MockLayer inside a list, i.e., a list of MockLayer objects but with a single layer."""
        return [self]

    def get_output_receptive_field(self, input_receptive_field, input_accumulated_stride):
        """Computes the receptive field of the layer and updates the accumulated stride.

        :param input_receptive_field: (integer) receptive field of the previous layer.
        :param input_accumulated_stride: (integer) stride that has been accumulated so far
        :return:
            output_receptive_field: (integer) new receptive field.
            output_accumulated_stride: (integer) updated accumulated stride.
        """
        kernel_term = (self.kernel_size - 1) * self.dilation_rate * input_accumulated_stride
        output_receptive_field = kernel_term + input_receptive_field
        output_accumulated_stride = input_accumulated_stride * self.stride
        return output_receptive_field, output_accumulated_stride


class MockConv(MockLayer):
    """Mocking convolutional layer."""
    def __init__(self, kernel_size, stride=1, dilation_rate=1, name=''):
        """
        :param kernel_size: (integer) size of the kernel.
        :param stride: (integer) size of the stride (it defaults to 1).
        :param dilation_rate: (integer) the dilation rate of the kernel (it defaults to 1).
        :param name: (string) an optional name of the layer (it defaults to '').
        """
        super(MockConv, self).__init__(kernel_size, stride, dilation_rate, name)


class MockPool(MockLayer):
    """Mocking pooling layer."""
    def __init__(self, pool_size, stride=None, name=''):
        """
        :param pool_size: (integer) pooling size of the layer (analogous to the kernel size).
        :param stride: (integer or None) size of the stride (it defaults to None which makes stride=pool_size).
        :param name: (string) an optional name of the layer (it defaults to '').
        """
        stride = pool_size if stride is None else stride
        super(MockPool, self).__init__(pool_size, stride, 1, name)


class MockModel(object):
    """A mocking model that tracks a list of mocking layers."""
    def __init__(self, initial_layer_list=None):
        """
        :param initial_layer_list: (list of MockLayer objects) an optional list of mocking layers
            that can be provided when initializing the model. It defaults to None, which initializes
            the model with an empty list of layers. Whatever your decision, you can stack more layers
            afterwards using the "add" method.
        """
        self.layers = [] if initial_layer_list is None else initial_layer_list

    def get_as_list(self):
        """Returns the list of the MockLayer objects contained in this MockModel."""
        return self.layers

    def add(self, new_block):
        """Stacks a new layer or a new group of layers to the model.

        :param new_block: (MockLayer or MockModel) the block to be appended (either a mocking layer
            or a group of mocking layers contained in a mocking model)
        """
        list_of_layers = new_block.get_as_list()
        self.layers.extend(list_of_layers)

    def get_layer_names(self):
        """Returns the names of the layers of the model."""
        return [layer.name for layer in self.layers]

    def summary(self):
        """Prints the names and parameters of the layers of the model."""
        print('Layer idx <name> (kernel size, stride, dilation rate)')
        print("-----------------------------------------------------")
        for i, layer in enumerate(self.layers):
            msg = 'Layer {} {} (k{}, s{}, d{}) '.format(
                i + 1, layer.name.ljust(20),
                layer.kernel_size, layer.stride, layer.dilation_rate
            )
            print(msg)
        print("")

    def get_receptive_field(self, verbose=False):
        """Computes the receptive field of the model.

        :param verbose: (boolean) whether you want to print each computation step (it defaults to False).
        :return:
            output_receptive_field: (integer) the receptive field of the last layer of the model.
            all_receptive_fields: (list of integers) all the intermediate receptive fields of the model.
        """
        output_receptive_field = 1
        accumulated_stride = 1
        all_receptive_fields = []
        for i, layer in enumerate(self.layers):
            output_receptive_field, accumulated_stride = layer.get_output_receptive_field(
                output_receptive_field, accumulated_stride)
            all_receptive_fields.append(output_receptive_field)
            if verbose:
                msg = 'Layer {} {} (k{}, s{}, d{}) Receptive field {}'.format(
                    i + 1, layer.name.ljust(20),
                    layer.kernel_size, layer.stride, layer.dilation_rate,
                    output_receptive_field
                )
                print(msg)
        if verbose:
            print("")
        return output_receptive_field, all_receptive_fields
