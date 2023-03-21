#ifndef NET_H_INCLUDED__
#define NET_H_INCLUDED__

#include <cstdint>

namespace cc0
{
	/// @brief Contains functions and structures that are exclusively meant for use within the net package.
	namespace net_internal
	{
		/// @brief A simple memory buffer that can allocate and automatically deallocate memory. Not for external use.
		/// @tparam type_t The type of the buffer.
		/// @note This is for use only within the net package. May very well change drastically or be removed on a whim.
		template < typename type_t >
		class buffer
		{
		private:
			type_t   *m_values;
			uint64_t  m_size;

		public:
			/// @brief Default constructor. Everything zeroed.
			buffer( void );

			/// @brief Creates a buffer of a given size.
			/// @param size The requested size of the buffer.
			explicit buffer(uint64_t size);

			/// @brief Deallocates memory.
			~buffer( void );

			/// @brief Returns a pointer to the underlying memory for access.
			/// @return A pointer to the underlying memory.
			operator type_t*( void );

			/// @brief Returns a pointer to the underlying memory for access.
			/// @return A pointer to the underlying memory.
			operator const type_t*( void ) const;

			/// @brief Deletes the old memory and allocates new given the requested size.
			/// @param size The requested size of the new buffer.
			void create(uint64_t size);

			/// @brief Deletes the memory and zeroes the buffer internals.
			void destroy( void );

			/// @brief Sets all values in buffer to the input value.
			/// @param val The input value to set all elements in the buffer.
			void set_vals(const type_t &val);

			/// @brief Gets the current number of elements (size) in the buffer.
			/// @return The size of the buffer.
			uint64_t size( void ) const;
		};
	}

	/// @brief Contains some default components that can be swapped out for custom ones in the neural network.
	namespace common
	{
		/// @brief Contains some default transfer functions and their derivatives.

		namespace transfer
		{
			/// @brief The fast sigmoid transfer function.
			/// @param x The input.
			/// @return The output.
			/// @sa d_fsig
			float fsig(float x);
			
			/// @brief Derive the output from the fast sigmoid transfer function.
			/// @param y The input (output from fsig).
			/// @return The output.
			/// @sa fsig
			float d_fsig(float y);

			/// @brief Trigonometric transfer function.
			/// @param x The input.
			/// @return The output.
			/// @sa d_tanh
			float tanh(float x);

			/// @brief Derive the output from the trigionometric transfer function.
			/// @param y The input (output from tanh).
			/// @return The output.
			/// @sa tanh
			float d_tanh(float y);
		}

		/// @brief Generate a random number between 0 rand RAND_MAX.
		/// @return A random number between 0 rand RAND_MAX.
		uint32_t random_u( void );
		
		/// @brief Generate a random number between 0-1.
		/// @return A random number between 0-1.
		float random_f( void );
	}

	/// @brief The neural network.
	class net
	{
	public:
		/// @brief A layer in the neural network.
		class layer
		{
		private:
			float    *m_neurons;
			float    *m_gradients;
			float    *m_weights;
			float    *m_delta_weights;
			uint64_t  m_neuron_count;
			uint64_t  m_weights_per_neuron;
			layer    *m_prev_layer;
			layer    *m_next_layer;

		public:
			/// @brief Sanity checks that memory does not overlap each other due to wrong assumptions/implementation.
			/// @return True if there is no overlap and memory is within allocated range.
			bool mem_ok( void ) const;

		private:
			/// @brief Do a feed forward to the next layer using the current neuron value and its weight towards the neurons in the next layer.
			/// @param neuron The value of the neuron. 
			/// @param weights The weights from the current neuron to each neuron in the next layer.
			/// @param next The next layer in the neuron network.
			static void feed_forward(float neuron, const float *weights, layer &next);

			/// @brief Calculates the gradients in this layer, and all previous layers, based off of the expected outputs.
			/// @param expected_outputs The expected outputs.
			/// @param transfer_derived_fn The derived version of the transfer function.
			void update_gradients(const float *expected_outputs, float (*transfer_derived_fn)(float));

			/// @brief Calculates the gradients in the output layer based off of the expected outputs.
			/// @param expected_outputs The expected outputs.
			/// @param transfer_derived_fn The derived version of the transfer function.
			/// @note Only use this function if this layer is the output (last) layer.
			void calculate_output_gradients(const float *expected_outputs, float (*transfer_derived_fn)(float));

			/// @brief Calculates the gradient based on the target value.
			/// @param neuron The actual neuron value.
			/// @param target The target neuron value.
			/// @param transfer_derived_fn The derived version of the transfer function.
			/// @return The gradient value.
			static float calculate_output_gradient(float neuron, float target, float (*transfer_derived_fn)(float));

			/// @brief Calculates the DOW.
			/// @param weights The weights from the current neuron to the next layer.
			/// @param next_layer The next layer.
			/// @return The DOW.
			static float sum_dow(const float *weights, const layer &next_layer);

			/// @brief Calculates the gradients in the hidden layer based off of the expected values in the next layer.
			/// @param next_layer The next layer.
			/// @param transfer_derived_fn The derived version of the transfer function.
			/// @note Only use this function if this layer is a hidden layer.
			void calculate_hidden_gradients(const layer &next_layer, float (*transfer_derived_fn)(float));

			/// @brief Calculates the gradient based on the target values in the next layer.
			/// @param neuron The actual neuron value.
			/// @param weights The weights from the neuron to the next layer.
			/// @param next_layer The target values in the next layer.
			/// @param transfer_derived_fn The derived version of the transfer function.
			/// @return The gradient value.
			static float calculate_hidden_gradient(float neuron, const float *weights, const layer &next_layer, float (*transfer_derived_fn)(float));

			/// @brief Updates the connection weights in the previous layer to this layer.
			void update_weights( void );

			/// @brief Updates the connection weights in the previous layer to this layer.
			/// @param prev_layer The previous layer. 
			void update_weights(layer &prev_layer);

			/// @brief Updates the connection weights in the previous layer to this layer.
			/// @param neuron_index The index of the neuron in the current layer.
			/// @param prev_layer The previous layer.
			void update_weights(uint64_t neuron_index, layer &prev_layer) const;

		public:
			/// @brief Default constructor. Everything zeroed.
			layer( void );

			/// @brief Sets the memory to be used within the layer as neuron data.
			/// @param memory The memory that will be used by the layer as neuron data, bias, and weights. Note that one extra value added on top of neuron_count will be used as bias.
			/// @param neuron_count The size of the input memory, for a total of neuron_count+1 including bias.
			/// @param weights_per_neuron The number of weights per neuron in the input memory. The number of weights must correspond to the number of neurons in the next layer.
			/// @param rand_f_fn The random function used to initiate weights to random values.
			layer(float *memory, uint64_t neuron_count, uint64_t weights_per_neuron, layer *prev_layer, layer *next_layer, float (*rand_f_fn)() = cc0::common::random_f);

			/// @brief Returns the data for the neurons.
			/// @return The data for the neurons.
			float *get_neurons( void );

			/// @brief Returns the data for the neurons.
			/// @return The data for the neurons.
			const float *get_neurons( void ) const;

			/// @brief Returns the data for the gradients.
			/// @return The data for the gradients.
			float *get_gradients( void );

			/// @brief Returns the data for the gradients.
			/// @return The data for the gradients.
			const float *get_gradients( void ) const;

			/// @brief Get the bias neuron.
			/// @return The bias neuron.
			float &get_bias( void );

			/// @brief Get the bias neuron.
			/// @return The bias neuron.
			float get_bias( void ) const;

			/// @brief Gets the number of neurons (size) in the layer.
			/// @return The number of neurons in the layer.
			uint64_t get_neuron_count( void ) const;

			/// @brief The number of gradients in this layer. Always the same as the number of neurons with the bias.
			/// @return The number of gradients in this layer.
			uint64_t get_gradient_count( void ) const;

			/// @brief Returns the number of biases. Only ever 1 or 0.
			/// @return 1 if there is a bias. 0 if there is no bias.
			uint64_t get_bias_count( void ) const;

			/// @brief Gets the number of neurons including the bias (size+1) in the layer.
			/// @return The number of neurons including the bias in the layer.
			/// @note The output layer does not have a bias.
			uint64_t get_neuron_count_with_bias( void ) const;

			/// @brief Returns the data to the weights for a given neuron.
			/// @param neuron_index The index of the neuron for which to get the weights.
			/// @return The weights.
			float *get_weights(uint64_t neuron_index);

			/// @brief Returns the data to the weights for a given neuron.
			/// @param neuron_index The index of the neuron for which to get the weights.
			/// @return The weights.
			const float *get_weights(uint64_t neuron_index) const;

			/// @brief Returns the data for the rate of change for the weights in the corresponding index.
			/// @return The data for the rate of change for the weights in the corresponding index.
			float *get_delta_weights(uint64_t neuron_index);

			/// @brief Returns the data for the rate of change for the weights in the corresponding index.
			/// @return The data for the rate of change for the weights in the corresponding index.
			const float *get_delta_weights(uint64_t neuron_index) const;

			/// @brief Returns the weights for the bias.
			/// @return The weights for  the bias.
			float *get_bias_weights( void );

			/// @brief Returns the weights for the bias.
			/// @return The weights for  the bias.
			const float *get_bias_weights( void ) const;

			/// @brief Returns the number of weight arrays available.
			/// @return The number of weight arrays in the layer.
			/// @note This value also applies for delta weights.
			uint64_t get_weight_array_count( void ) const;

			/// @brief Gets the number of weights per neurons.
			/// @return The number of weights per neuron.
			uint64_t get_weights_per_neuron( void ) const;

			/// @brief Returns the total number of data points in the data array.
			/// @return The total number of data points in the data array. 
			uint64_t get_total_size( void ) const;

			/// @brief Feeds values forward to the next layer in the neural network.
			/// @param transfer_fn The transfer function to use.
			/// @note The number of neurons in the next layer must correspond to the number of weights allocated for the layer feeding values into the next layer.
			void feed_forward(float (*transfer_fn)(float)) const;

			/// @brief Calculates the memory usage (in number of elements) needed for a layer with the specified requirements.
			/// @param neuron_count The number of neurons to be used.
			/// @param weights_per_neuron The number of weights (corresponds to the number of neurons in the next layer, not including bias);
			/// @return The number of floating point variables needed for a layer with the specified requirements.
			static uint64_t calculate_memory_usage(uint64_t neuron_count, uint64_t weights_per_neuron);

			/// @brief Updates gradients in this layer and all previous layers, then updates the weights in this layer and all previous layers.
			/// @param expected_outputs The expected outputs.
			/// @param transfer_derived_fn The derived version of the transfer function.
			void propagate_backward(const float *expected_outputs, float (*transfer_derived_fn)(float));

			/// @brief Determines if this layer has a bias and weights.
			/// @return True if this layer has bias and weights components.
			/// @note A layer can not have a bias and no weights, or weights and no bias. A layer always either has both, or neither.
			/// @note Generally only the output layer does not have a bias and weights. However, there is nothing preventing the user from initiating a non-output layer as having no weights per neuron, and thus no bias either.
			bool has_bias_and_weights( void ) const;

			/// @brief Determines if this layer is the first layer in the network.
			/// @return True if this layer is the first layer in the network.
			bool is_input_layer( void ) const;

			/// @brief Determines if this layer is the last layer in the network.
			/// @return True if this layer is the last layer in the network.
			bool is_output_layer( void ) const;
		};

	private:
		net_internal::buffer<float>  m_buffer;                        // The main memory buffer.
		net_internal::buffer<layer>  m_layers;                        // The buffer containing layers.
		float                        (*m_transfer_fn)(float);         // The transfer function.
		float                        (*m_transfer_derived_fn)(float); // The derived version of the transfer function.
		float                        m_error;                         // The total error for the last iteration.
		float                        m_average_error;                 // Total error for the last iterations defined by error series count.
		uint64_t                     m_error_series_count;            // The number of iterations to smooth the average error over.

	private:
		/// @brief  Returns the output layer for access.
		/// @return The output layer.
		layer &get_output_layer_rw( void );

		/// @brief  Returns the input layer for access.
		/// @return The input layer.
		layer &get_input_layer_rw( void );

		/// @brief Updates the error.
		/// @param layer The layer to generate the error for.
		/// @param expected_outputs The expected outputs in the given layer.
		/// @note The number of elements in expected_outputs must match the number of neurons in the layer (excluding bias).
		void update_error(const layer &layer, const float *expected_outputs);

	public:
		/// @brief Default contructor. Everything zeroed.
		net( void );

		/// @brief Allocates memory within the neural network given the input sizes.
		/// @param topography The number of neurons to allocate for each layer in the network.
		/// @param num_layers The number of layers in the topography. Must be at least 2.
		/// @param random_fn The function used to initially randomize the layer data.
		net(const uint32_t *topography, uint32_t num_layers, float (*random_fn)() = common::random_f);

		/// @brief Allocates memory within the neural network given the input sizes.
		/// @tparam num_layers The number of layers in the topography. Must be at least 2. 
		/// @param topography The number of neurons to allocate for each layer in the network.
		/// @param random_fn The function used to initially randomize the layer data.
		template < uint64_t num_layers >
		net(const uint32_t (&topography)[num_layers], float (*random_fn)() = common::random_f);

		/// @brief Copies a neural network.
		/// @param n The neural network to copy.
		net(const net &n);

		/// @brief Allocates memory within the neural network given the input sizes.
		/// @param topography The number of neurons to allocate for each layer in the network.
		/// @param num_layers The number of layers in the topography. Must be at least 2.
		/// @param random_fn The function used to initially randomize the layer data.
		/// @note If num_layers is less than 2, the current net will be destroyed.
		void create(const uint32_t *topography, uint32_t num_layers, float (*random_fn)() = common::random_f);

		/// @brief Allocates memory within the neural network given the input sizes.
		/// @tparam num_layers The number of layers in the topography. Must be at least 2.
		/// @param topography The number of neurons to allocate for each layer in the network.
		/// @param random_fn The function used to initially randomize the layer data.
		/// @note If num_layers is less than 2, the current net will be destroyed.
		template < uint64_t num_layers >
		void create(const uint32_t (&topography)[num_layers], float (*random_fn)() = common::random_f);

		/// @brief Deallocates all memory.
		void destroy( void );

		/// @brief Trains the neural network by comparing the results in the output layer to the expected results and optimizing the weights and biases so that the results will more closely approximate the expected outputs next time.
		/// @param inputs An array of values to be used in the input layer of the network. The number of values in the parameter array must correspond to the number of neurons allocated for the input layer.
		/// @param expected_outputs Array of values that tries to tell the neural network if the results in the output layer matches the expected results. If not, the neural net will attempt to adjust its weights and biases to give a more accurate approximation during next run. The number of values in the parameter array must correspond to the number of neurons allocated for the output layer.
		/// @return The error for the current iteration of training.
		/// @sa feed_forward
		/// @sa propagate_backward
		float train(const float *inputs, const float *expected_outputs);

		/// @brief Produce results in the output layer given inputs in the input layer without performing a training step.
		/// @param inputs An array of values to be used in the input layer of the network. The number of values in the parameter array must correspond to the number of neurons allocated for the input layer.
		/// @sa train
		/// @sa propagate_backward
		void feed_forward(const float *inputs);

		/// @brief Refines neural net internals when the expected output is known in order to arrive at a better result feeding forward next time. Call this after feed_forward.
		/// @param expected_outputs Array of values that tries to tell the neural network if the results in the output layer matches the expected results. If not, the neural net will attempt to adjust its weights and biases to give a more accurate approximation during next run. The number of values in the parameter array must correspond to the number of neurons allocated for the output layer.
		/// @return The error for the session.
		/// @sa train
		/// @sa feed_forward
		float propagate_backward(const float *expected_outputs);

		/// @brief Returns the number of layers in the neural network, including the input and output layers.
		/// @return The number of layers in the neural network.
		uint64_t get_layer_count( void ) const;

		/// @brief Returns a layer for access.
		/// @param index The index of the layer to access.
		/// @return A layer for access.
		/// @sa get_layer_count
		const layer &get_layer(uint64_t index) const;

		/// @brief  Returns the output layer for access.
		/// @return The output layer.
		const layer &get_output_layer( void ) const;

		/// @brief  Returns the input layer for access.
		/// @return The input layer.
		const layer &get_input_layer( void ) const;

		/// @brief Set custom transfer functions.
		/// @param transfer_fn The main transfer function, transforming input into output.
		/// @param transfer_derived_fn The derived transfer function, transforming input (transfer function output) into output.
		/// @note Transfer functions are set to sensible defaults. There is no need to set these to other than default. 
		void set_transfer_functions(float (*transfer_fn)(float), float (*transfer_derived_fn)(float));

		/// @brief Returns the recent average error. Uses the error series count to smooth the results.
		/// @return The recent average error.
		/// @sa set_error_series_count
		float get_average_error( void ) const;

		/// @brief Sets the number of iterations to smooth the error results over to get a less noisy output of the error over time.
		/// @param count The number of iterations to smooth the error results over.
		void set_error_series_count(uint64_t count);

		/// @brief Creates a new neural network given randomly selected weights and biases from two neural networks of the same topography.
		/// @param a A neural net. 
		/// @param b Another neural net.
		/// @return The resulting, spliced neural network. If the input networks are not same topography, an empty neural network is returned.
		static net splice(const net &a, const net &b, uint32_t (*rand_u_fn)() = common::random_u);
	};
}

template < typename type_t >
cc0::net_internal::buffer<type_t>::buffer( void ) : m_values(nullptr), m_size(0)
{}

template < typename type_t >
cc0::net_internal::buffer<type_t>::buffer(uint64_t size) : buffer()
{
	create(size);
}

template < typename type_t >
cc0::net_internal::buffer<type_t>::~buffer( void )
{
	destroy();
}

template < typename type_t >
cc0::net_internal::buffer<type_t>::operator type_t*( void )
{
	return m_values;
}

template < typename type_t >
cc0::net_internal::buffer<type_t>::operator const type_t*( void ) const
{
	return m_values;
}

template < typename type_t >
void cc0::net_internal::buffer<type_t>::create(uint64_t size)
{
	destroy();
	if (size > 0) {
		m_values = new type_t[size];
		m_size = size;
	}
}

template < typename type_t >
void cc0::net_internal::buffer<type_t>::destroy( void )
{
	delete [] m_values;
	m_size = 0;
}

template < typename type_t >
void cc0::net_internal::buffer<type_t>::set_vals(const type_t &val)
{
	for (uint64_t i = 0; i < m_size; ++i) {
		m_values[i] = val;
	}
}

template < typename type_t >
uint64_t cc0::net_internal::buffer<type_t>::size( void ) const
{
	return m_size;
}

template < uint64_t num_layers >
cc0::net::net(const uint32_t (&topography)[num_layers], float (*random_fn)()) : net(topography, num_layers, random_fn)
{}

template < uint64_t num_layers >
void cc0::net::create(const uint32_t (&topography)[num_layers], float (*random_fn)())
{
	create(topography, num_layers, random_fn);
}

#endif
