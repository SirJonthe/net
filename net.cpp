#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "net.h"

void cc0::net::layer::feed_forward(float neuron, const float *weights, layer &next)
{
	for (uint64_t i = 0; i < next.get_neuron_count(); ++i) {
		next.get_neurons()[i] += neuron * weights[i];
	}
}

void cc0::net::layer::calculate_output_gradients(const float *expected_outputs, float (*transfer_derived_fn)(float))
{
	for (uint64_t i = 0; i < get_neuron_count(); ++i) {
		get_gradients()[i] = calculate_output_gradient(get_neurons()[i], expected_outputs[i], transfer_derived_fn);
	}
}

float cc0::net::layer::calculate_output_gradient(float neuron, float target, float (*transfer_derived_fn)(float))
{
	return (target - neuron) * transfer_derived_fn(neuron);
}

float cc0::net::layer::sum_dow(const float *weights, const cc0::net::layer &next_layer)
{
	float sum = 0.0f;
	for (uint64_t i = 0; i < next_layer.get_neuron_count(); ++i) {
		sum += weights[i] * next_layer.get_gradients()[i];
	}
	return sum;
}

void cc0::net::layer::calculate_hidden_gradients(const cc0::net::layer &next_layer, float (*transfer_derived_fn)(float))
{
	if (this != nullptr) {
		for (uint64_t i = 0; i < get_neuron_count_with_bias(); ++i) {
			get_gradients()[i] = calculate_hidden_gradient(get_neurons()[i], get_weights(i), next_layer, transfer_derived_fn);
		}
	}
}

float cc0::net::layer::calculate_hidden_gradient(float neuron, const float *weights, const cc0::net::layer &next_layer, float (*transfer_derived_fn)(float))
{
	return sum_dow(weights, next_layer) * transfer_derived_fn(neuron);
}

void cc0::net::layer::update_weights(cc0::net::layer &prev_layer)
{
	for (uint64_t i = 0; i < get_neuron_count(); ++i) {
		update_weights(i, prev_layer);
	}
}

static constexpr float ETA = 0.15f; /// @brief Tunable parameter ETA. 0.1 to 0.2 is good.
static constexpr float ALPHA = 0.5f; /// @brief Tunable parameter ALPHA. 0.0 to 1.0 is good.

void cc0::net::layer::update_weights(uint64_t neuron_index, cc0::net::layer &prev_layer) const
{
	// TODO This should be the other way around, i.e. the next layer is modifying this layer. That way we do not need indexes.
	const float neuron = get_neurons()[neuron_index];
	const float gradient = get_gradients()[neuron_index];
	for (uint64_t i = 0; i < prev_layer.get_neuron_count_with_bias(); ++i) {
		prev_layer.get_delta_weights(i)[neuron_index] =
			ETA * neuron * gradient +
			ALPHA * prev_layer.get_delta_weights(i)[neuron_index];
		prev_layer.get_weights(i)[neuron_index] += prev_layer.get_delta_weights(i)[neuron_index]; // TODO I do not think we need to save the delta_weight. Just assign the delta here...
	}
}

cc0::net::layer::layer( void ) : m_neurons(nullptr), m_neuron_count(0), m_weights_per_neuron(0), m_prev_layer(nullptr), m_next_layer(nullptr)
{}

cc0::net::layer::layer(float *memory, uint64_t neuron_count, uint64_t weights_per_neuron, cc0::net::layer *prev_layer, cc0::net::layer *next_layer, float (*rand_fn)()) : m_neurons(memory), m_neuron_count(neuron_count), m_weights_per_neuron(weights_per_neuron), m_prev_layer(prev_layer), m_next_layer(next_layer)
{
	for (uint64_t n = 0; n < get_neuron_count_with_bias(); ++n) {
		float *w = get_weights(n);
		for (uint64_t i = 0; i < m_weights_per_neuron; ++i) {
			w[i] = rand_fn();
		}
	}
}

float *cc0::net::layer::get_neurons( void )
{
	return m_neurons;
}

const float *cc0::net::layer::get_neurons( void ) const
{
	return m_neurons;
}

float *cc0::net::layer::get_gradients( void )
{
	return get_weights(get_neuron_count());
}

const float *cc0::net::layer::get_gradients( void ) const
{
	return get_weights(get_neuron_count());
}

float &cc0::net::layer::get_bias( void )
{
	return m_neurons[m_neuron_count];
}

float cc0::net::layer::get_bias( void ) const
{
	return m_neurons[m_neuron_count];
}

uint64_t cc0::net::layer::get_neuron_count( void ) const
{
	return m_neuron_count;
}

uint64_t cc0::net::layer::get_gradient_count( void ) const
{
	return get_neuron_count_with_bias();
}

uint64_t cc0::net::layer::get_neuron_count_with_bias( void ) const
{
	return m_neuron_count + 1;
}

float *cc0::net::layer::get_weights(uint64_t neuron_index)
{
	return m_neurons + get_neuron_count_with_bias() + m_weights_per_neuron * neuron_index;
}

const float *cc0::net::layer::get_weights(uint64_t neuron_index) const
{
	return m_neurons + get_neuron_count_with_bias() + m_weights_per_neuron * neuron_index;
}

float *cc0::net::layer::get_delta_weights(uint64_t neuron_index)
{
	return get_weights(get_neuron_count_with_bias()) + m_weights_per_neuron * neuron_index;
}

const float *cc0::net::layer::get_delta_weights(uint64_t neuron_index) const
{
	return get_weights(get_neuron_count_with_bias()) + m_weights_per_neuron * neuron_index;
}

float *cc0::net::layer::get_bias_weights( void )
{
	return get_weights(m_neuron_count);
}

const float *cc0::net::layer::get_bias_weights( void ) const
{
	return get_weights(m_neuron_count);
}

uint64_t cc0::net::layer::get_weights_per_neuron( void ) const
{
	return m_weights_per_neuron;
}

uint64_t cc0::net::layer::get_total_size( void ) const
{
	return calculate_memory_usage(m_neuron_count, m_weights_per_neuron);
}

void cc0::net::layer::feed_forward(float (*transfer_fn)(float)) const
{
	if (m_next_layer != nullptr) {
		for (uint64_t i = 0; i < m_next_layer->get_neuron_count(); ++i) {
			m_next_layer->get_neurons()[i] = 0.0f;
		}
		for (uint64_t i = 0; i < get_neuron_count_with_bias(); ++i) {
			feed_forward(m_neurons[i], get_weights(i), *m_next_layer);
		}
		for (uint64_t i = 0; i < m_next_layer->get_neuron_count(); ++i) {
			m_next_layer->get_neurons()[i] = transfer_fn(m_next_layer->get_neurons()[i]);
		}
		m_next_layer->feed_forward(transfer_fn);
	}
}

uint64_t cc0::net::layer::calculate_memory_usage(uint64_t neuron_count, uint64_t weights_per_neuron)
{
	return (neuron_count + 1) + neuron_count + (neuron_count + 1) * (weights_per_neuron) * 2;
}

void cc0::net::layer::update_gradients(const float *expected_outputs, float (*transfer_derived_fn)(float))
{
	if (this != nullptr) {
		calculate_output_gradients(expected_outputs, transfer_derived_fn);
		m_prev_layer->calculate_hidden_gradients(*this, transfer_derived_fn);
	}
}

void cc0::net::layer::update_weights( void )
{
	if (this != nullptr) {
		if (m_next_layer != nullptr) {
			m_next_layer->update_weights(*this);
		}
		m_prev_layer->update_weights();
	}
}

void cc0::net::layer::propagate_backward(const float *expected_outputs, float (*transfer_derived_fn)(float))
{
	update_gradients(expected_outputs, transfer_derived_fn);
	update_weights();
}

static bool init_default_rand( void )
{
	srand(time(nullptr));
	return true;
}

float cc0::common::random_unit( void )
{
	static const bool seed_init = init_default_rand();
	static constexpr float rand_weight = 1.0f / RAND_MAX;
	return rand() * rand_weight;
}

float cc0::common::fast_sigmoid(float x)
{
	return x / (1.0f + std::abs(x));
}

float cc0::common::derive_fast_sigmoid(float y)
{
	return y * (1.0f - y);
}

cc0::net::layer &cc0::net::get_output_layer( void )
{
	return m_layers[m_layers.size() - 1];
}

cc0::net::layer &cc0::net::get_input_layer( void )
{
	return m_layers[0];
}

void cc0::net::update_error(const layer &out, const float *expected_outputs)
{
	m_error = 0.0f;
	for (uint64_t i = 0; i < out.get_neuron_count(); ++i) {
		const float delta = expected_outputs[i] - out.get_neurons()[i];
		m_error += delta * delta;
	}
	m_error = sqrtf(m_error / out.get_neuron_count());
}

cc0::net::net( void ) : m_buffer(), m_layers(), m_transfer_fn(common::fast_sigmoid), m_transfer_derived_fn(common::derive_fast_sigmoid), m_error(0.0f)
{}

cc0::net::net(const uint32_t *topography, uint32_t num_layers, float (*random_fn)()) : net()
{
	create(topography, num_layers, random_fn);
}
#include <iostream>
void cc0::net::create(const uint32_t *topography, uint32_t num_layers, float (*random_fn)())
{
	assert(num_layers >= 2);

	uint64_t total_memory = 0;
	for (uint64_t i = 0; i < num_layers; ++i) {
		total_memory += layer::calculate_memory_usage(
			topography[i],
			i < num_layers - 1 ? topography[i + 1] : 0
		);
	}
	m_buffer.create(total_memory);
	m_layers.create(num_layers);

	float *ptr = m_buffer;
	if (ptr != nullptr) {
		for (uint32_t i = 0; i < num_layers; ++i) {
			m_layers[i] = layer(
				ptr,
				topography[i], i < num_layers - 1 ? topography[i + 1] : 0,
				i == 0 ? nullptr : &m_layers[i - 1],
				i == num_layers-1 ? nullptr : &m_layers[i + 1],
				random_fn
			);
			ptr += m_layers[i].get_total_size();
		}
		assert(ptr == m_buffer + total_memory);
	}
}

void cc0::net::destroy( void )
{
	m_buffer.destroy();
	m_layers.destroy();
}

void cc0::net::train(const float *inputs, const float *expected_outputs)
{
	feed_forward(inputs);
	propagate_backward(expected_outputs);
}

void cc0::net::feed_forward(const float *inputs)
{
	if (m_layers.size() > 0) {
		layer &in = get_input_layer();
		for (uint64_t i = 0; i < in.get_neuron_count(); ++i) {
			in.get_neurons()[i] = inputs[i];
		}
		in.feed_forward(m_transfer_fn);
	}
}

void cc0::net::propagate_backward(const float *expected_outputs)
{
	if (m_layers.size() > 0) {
		layer &out = get_output_layer();
		update_error(out, expected_outputs);
		out.propagate_backward(expected_outputs, m_transfer_derived_fn);
	}
}

uint64_t cc0::net::get_layer_count( void ) const
{
	return m_layers.size();
}

const cc0::net::layer &cc0::net::get_layer(uint64_t index) const
{
	return m_layers[index];
}

const cc0::net::layer &cc0::net::get_output_layer( void ) const
{
	return m_layers[m_layers.size() - 1];
}

const cc0::net::layer &cc0::net::get_input_layer( void ) const
{
	return m_layers[0];
}

void cc0::net::set_transfer_functions(float (*transfer_fn)(float), float (*transfer_derived_fn)(float))
{
	m_transfer_fn = transfer_fn;
	m_transfer_derived_fn = transfer_derived_fn;
}

float cc0::net::get_error( void ) const
{
	return m_error;
}
