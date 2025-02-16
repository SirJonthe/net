/// @file net.cpp
/// @brief Contains a minimalist library that implements a fully connected neural network.
/// @author github.com/SirJonthe
/// @date 2022, 2023
/// @copyright Public domain.
/// @license CC0 1.0

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "net.h"

bool cc0::net::layer::mem_ok( void ) const
{
	struct seg
	{
		const float *a; // memory start, inclusive
		const float *b; // memory end, inclusive
		const char *name;
	};

	const seg full = { // The entire memory consumption must exactly fit here.
		m_neurons,
		m_neurons + calculate_memory_usage(m_neuron_count, m_weights_per_neuron) - 1,
		"full"
	};

	net_internal::buffer<seg> segments;
	segments.create(
		1 + // one segment of neurons
		1 + // one segment of gradients
		get_weight_array_count() + // segments of weights
		get_weight_array_count() // segments of delta weights
	);

	segments[0] = {
		get_neurons(),
		get_neurons() + get_neuron_count_with_bias() - 1,
		"neurons"
	};
	segments[1] = {
		get_gradients(),
		get_gradients() + get_gradient_count() - 1,
		"gradients"
	};

	for (uint64_t i = 0; i < get_weight_array_count(); ++i) {
		segments[i + 2] = {
			get_weights(i),
			get_weights(i) + get_weights_per_neuron() - 1,
			"weights"
		};
		segments[i + get_weight_array_count() + 2] = {
			get_delta_weights(i),
			get_delta_weights(i) + get_weights_per_neuron() - 1,
			"delta weights"
		};
	}

	uint64_t overrides = 0;

	seg utilized = segments[segments.size()-1];
	for (uint64_t i = 0; i < segments.size() - 1; ++i) {
		seg a = segments[i];
		utilized.a = utilized.a < a.a ? utilized.a : a.a;
		utilized.b = utilized.b > a.b ? utilized.b : a.b;
		for (uint64_t j = i + 1; j < segments.size(); ++j) {
			seg b = segments[j];
			overrides += (( a.a >= b.a && a.a <= b.b ) || ( a.b >= b.a && a.b <= b.b ));
			overrides += (( b.a >= a.a && b.a <= a.b ) || ( b.b >= a.a && b.b <= a.b ));
		}
	}

	overrides += !(utilized.a >= full.a && utilized.b <= full.b);
	overrides += !(utilized.a == full.a && utilized.b == full.b);

	const int64_t diff = (utilized.b - utilized.a) - (full.b - full.a);

	return overrides == 0 && diff == 0;
}

void cc0::net::layer::feed_forward(float neuron, const float *weights, layer &next)
{
	for (uint64_t i = 0; i < next.get_neuron_count(); ++i) {
		next.get_neurons()[i] += neuron * weights[i];
	}
}

void cc0::net::layer::update_gradients(const float *expected_outputs, float (*transfer_derived_fn)(float))
{
	if (this != nullptr) {
		calculate_output_gradients(expected_outputs, transfer_derived_fn);
		m_prev_layer->calculate_hidden_gradients(*this, transfer_derived_fn);
	}
}

void cc0::net::layer::calculate_output_gradients(const float *expected_outputs, float (*transfer_derived_fn)(float))
{
	float *g = get_gradients();
	const float *n = get_neurons();
	for (uint64_t i = 0; i < get_neuron_count(); ++i) {
		g[i] = calculate_output_gradient(n[i], expected_outputs[i], transfer_derived_fn);
	}
}

float cc0::net::layer::calculate_output_gradient(float neuron, float target, float (*transfer_derived_fn)(float))
{
	return (target - neuron) * transfer_derived_fn(neuron);
}

float cc0::net::layer::sum_dow(const float *weights, const cc0::net::layer &next_layer)
{
	float sum = 0.0f;
	const float *g = next_layer.get_gradients();
	for (uint64_t i = 0; i < next_layer.get_neuron_count(); ++i) {
		sum += weights[i] * g[i];
	}
	return sum;
}

void cc0::net::layer::calculate_hidden_gradients(const cc0::net::layer &next_layer, float (*transfer_derived_fn)(float))
{
	if (this != nullptr) {
		float *g = get_gradients();
		const float *n = get_neurons();
		for (uint64_t i = 0; i < get_neuron_count_with_bias(); ++i) {
			g[i] = calculate_hidden_gradient(n[i], get_weights(i), next_layer, transfer_derived_fn);
		}
	}
}

float cc0::net::layer::calculate_hidden_gradient(float neuron, const float *weights, const cc0::net::layer &next_layer, float (*transfer_derived_fn)(float))
{
	return sum_dow(weights, next_layer) * transfer_derived_fn(neuron);
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
	const float g = get_gradients()[neuron_index];
	float *pn = prev_layer.get_neurons();
	for (uint64_t i = 0; i < prev_layer.get_neuron_count_with_bias(); ++i) {
		float *pd = prev_layer.get_delta_weights(i);
		pd[neuron_index] =
			ETA * pn[i] * g +
			ALPHA * pd[neuron_index]
		;
		prev_layer.get_weights(i)[neuron_index] += pd[neuron_index];
	}
}

cc0::net::layer::layer( void ) : m_neurons(nullptr), m_gradients(nullptr), m_weights(nullptr), m_delta_weights(nullptr), m_neuron_count(0), m_weights_per_neuron(0), m_prev_layer(nullptr), m_next_layer(nullptr)
{}

cc0::net::layer::layer(float *memory, uint64_t neuron_count, uint64_t weights_per_neuron, cc0::net::layer *prev_layer, cc0::net::layer *next_layer, float (*rand_f_fn)()) : 
	m_neurons(memory),
	m_gradients(nullptr),
	m_weights(nullptr),
	m_delta_weights(nullptr),
	m_neuron_count(neuron_count),
	m_weights_per_neuron(weights_per_neuron),
	m_prev_layer(prev_layer), m_next_layer(weights_per_neuron > 0 ? next_layer : nullptr)
{
	m_gradients = m_neurons + get_neuron_count_with_bias();
	if (has_bias_and_weights()) {
		m_weights = m_gradients + get_gradient_count();
		m_delta_weights = m_weights + get_weights_per_neuron() * get_weight_array_count();
		get_bias() = 1.0f;
	}
	for (uint64_t n = 0; n < get_weight_array_count(); ++n) {
		float *w = get_weights(n);
		float *d = get_delta_weights(n);
		for (uint64_t i = 0; i < get_weights_per_neuron(); ++i) {
			w[i] = rand_f_fn();
			d[i] = 0.0f;
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
	return m_gradients;
}

const float *cc0::net::layer::get_gradients( void ) const
{
	return m_gradients;
}

float &cc0::net::layer::get_bias( void )
{
	float *n = has_bias_and_weights() ? m_neurons : nullptr;
	return n[m_neuron_count];
}

float cc0::net::layer::get_bias( void ) const
{
	const float *n = has_bias_and_weights() ? m_neurons : nullptr;
	return n[m_neuron_count];
}

uint64_t cc0::net::layer::get_neuron_count( void ) const
{
	return m_neuron_count;
}

uint64_t cc0::net::layer::get_gradient_count( void ) const
{
	return get_neuron_count_with_bias();
}

uint64_t cc0::net::layer::get_bias_count( void ) const
{
	return has_bias_and_weights() ? 1 : 0;
}

uint64_t cc0::net::layer::get_neuron_count_with_bias( void ) const
{
	return m_neuron_count + get_bias_count();
}

float *cc0::net::layer::get_weights(uint64_t neuron_index)
{
	return
		has_bias_and_weights() ?
		m_weights + m_weights_per_neuron * neuron_index :
		nullptr;
}

const float *cc0::net::layer::get_weights(uint64_t neuron_index) const
{
	return
		has_bias_and_weights() ?
		m_weights + m_weights_per_neuron * neuron_index :
		nullptr;
}

float *cc0::net::layer::get_delta_weights(uint64_t neuron_index)
{
	return
		has_bias_and_weights() ?
		m_delta_weights + m_weights_per_neuron * neuron_index :
		nullptr;
}

const float *cc0::net::layer::get_delta_weights(uint64_t neuron_index) const
{
	return
		has_bias_and_weights() ?
		m_delta_weights + m_weights_per_neuron * neuron_index :
		nullptr;
}

float *cc0::net::layer::get_bias_weights( void )
{
	return get_weights(m_neuron_count);
}

const float *cc0::net::layer::get_bias_weights( void ) const
{
	return get_weights(m_neuron_count);
}

uint64_t cc0::net::layer::get_weight_array_count( void ) const
{
	return has_bias_and_weights() ? get_neuron_count_with_bias() : 0;
}

uint64_t cc0::net::layer::get_weights_per_neuron( void ) const
{
	return !is_output_layer() ? m_weights_per_neuron : 0;
}

uint64_t cc0::net::layer::get_total_size( void ) const
{
	return calculate_memory_usage(get_neuron_count(), get_weights_per_neuron());
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
	const uint64_t include_bias = weights_per_neuron > 0 ? 1 : 0;
	return
		(neuron_count + include_bias) +             // The number of neurons including the optional bias.
		(neuron_count + include_bias) +             // The number of gradients including the gradient for the optional bias.
		(neuron_count + 1) * (weights_per_neuron) + // The number of weights.
		(neuron_count + 1) * (weights_per_neuron);  // The number of delta weights.
}

void cc0::net::layer::propagate_backward(const float *expected_outputs, float (*transfer_derived_fn)(float))
{
	update_gradients(expected_outputs, transfer_derived_fn);
	update_weights();
}

bool cc0::net::layer::has_bias_and_weights( void ) const
{
	return get_weights_per_neuron() > 0;
}

bool cc0::net::layer::is_input_layer( void ) const
{
	return m_prev_layer == nullptr;
}

bool cc0::net::layer::is_output_layer( void ) const
{
	return m_next_layer == nullptr;
}

uint32_t cc0::common::random_u( void )
{
	return (uint32_t(rand()) << 1) & (uint32_t(rand()) & 1);
}

float cc0::common::random_f( void )
{
	//static const bool seed_init = init_default_rand();
	static constexpr float rand_weight = 1.0f / RAND_MAX;
	return rand() * rand_weight;
}

float cc0::common::transfer::fsig(float x)
{
	return x / (1.0f + std::abs(x));
}

float cc0::common::transfer::d_fsig(float y)
{
	return y * (1.0f - y);
}

float cc0::common::transfer::tanh(float x)
{
	return std::tanh(x);
}

float cc0::common::transfer::d_tanh(float y)
{
	return 1.0f - y * y;
}

cc0::net::layer &cc0::net::get_output_layer_rw( void )
{
	return m_layers[m_layers.size() - 1];
}

cc0::net::layer &cc0::net::get_input_layer_rw( void )
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

	m_average_error =
		(m_average_error * m_error_series_count + m_error) /
		(m_error_series_count + 1.0f);
}

cc0::net::net( void ) : m_buffer(), m_layers(), m_transfer_fn(common::transfer::tanh), m_transfer_derived_fn(common::transfer::d_tanh), m_error(0.0f), m_average_error(0.0f), m_error_series_count(100)
{}

cc0::net::net(const uint32_t *topography, uint32_t num_layers, float (*random_fn)()) : net()
{
	create(topography, num_layers, random_fn);
}

cc0::net::net(const cc0::net &n) : net()
{
	m_buffer.create(n.m_buffer.size());
	float *mem = m_buffer;
	m_layers.create(n.m_layers.size());
	for (uint64_t i = 0; i < m_layers.size(); ++i) {
		m_layers[i] = layer(
			mem,
			n.m_layers[i].get_neuron_count(),
			n.m_layers[i].get_weights_per_neuron(),
			i > 0 ? &m_layers[i-1] : nullptr,
			i < m_layers.size() - 1 ? &m_layers[i+1] : nullptr,
			nullptr // supply null to not initialize weights and biases
		);
		mem += layer::calculate_memory_usage(n.m_layers[i].get_neuron_count(), n.m_layers[i].get_weights_per_neuron());
	}
	for (uint64_t i = 0; i < m_buffer.size(); ++i) {
		m_buffer[i] = n.m_buffer[i];
	}
	m_error               = n.m_error;
	m_average_error       = n.m_average_error;
	m_error_series_count  = n.m_error_series_count;
	m_transfer_fn         = n.m_transfer_fn;
	m_transfer_derived_fn = n.m_transfer_derived_fn;
}

void cc0::net::create(const uint32_t *topography, uint32_t num_layers, float (*random_fn)())
{
	if (num_layers > 1) {
		uint64_t total_memory = 0;
		for (uint64_t i = 0; i < num_layers; ++i) {
			total_memory += layer::calculate_memory_usage(
				topography[i],
				i < num_layers - 1 ? topography[i + 1] : 0
			);
		}
		m_buffer.create(total_memory);
		m_buffer.set_vals(0.0f);
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
	} else {
		destroy();
	}
}

void cc0::net::destroy( void )
{
	m_buffer.destroy();
	m_layers.destroy();
}

float cc0::net::train(const float *inputs, const float *expected_outputs)
{
	feed_forward(inputs);
	return propagate_backward(expected_outputs);
}

void cc0::net::feed_forward(const float *inputs)
{
	if (m_layers.size() > 0) {
		layer &in = get_input_layer_rw();
		for (uint64_t i = 0; i < in.get_neuron_count(); ++i) {
			in.get_neurons()[i] = inputs[i];
		}
		in.feed_forward(m_transfer_fn);
	}
}

float cc0::net::propagate_backward(const float *expected_outputs)
{
	if (m_layers.size() > 0) {
		layer &out = get_output_layer_rw();
		update_error(out, expected_outputs);
		out.propagate_backward(expected_outputs, m_transfer_derived_fn);
	}
	return m_error;
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

float cc0::net::get_average_error( void ) const
{
	return m_average_error;
}

void cc0::net::set_error_series_count(uint64_t count)
{
	m_error_series_count = count > 0 ? count : 1;
}

template < typename type_t >
static const type_t &rselect(const type_t &a, const type_t &b, uint32_t (*rand_u_fn)())
{
	return (rand_u_fn() & 1) ? a : b;
}

cc0::net cc0::net::splice(const cc0::net &a, const cc0::net &b, uint32_t (*rand_u_fn)())
{
	cc0::net out;

	if (a.get_layer_count() != b.get_layer_count()) {
		return out;
	}
	
	for (uint64_t i = 0; i < a.get_layer_count(); ++i) {
		if (a.get_layer(i).get_neuron_count() != b.get_layer(i).get_neuron_count()) {
			return out;
		}
	}

	net_internal::buffer<uint32_t> topo(a.get_layer_count());
	for (uint64_t i = 0; i < a.get_layer_count(); ++i) {
		topo[i] = a.get_layer(i).get_neuron_count();
	}

	out.create(topo, topo.size());

	for (uint64_t l = 0; l < out.get_layer_count(); ++l) {
		for (uint64_t n = 0; n < out.m_layers[l].get_neuron_count_with_bias(); ++n) {
			const net &r = rselect(a, b, rand_u_fn);
			out.m_layers[l].get_neurons()[n] = r.m_layers[l].get_neurons()[n];
			out.m_layers[l].get_gradients()[n] = r.m_layers[l].get_gradients()[n];
		}
		for (uint64_t wa = 0; wa < out.m_layers[l].get_weight_array_count(); ++wa) {
			for (uint64_t w = 0; w < out.m_layers[l].get_weights_per_neuron(); ++w) {
				const net &r = rselect(a, b, rand_u_fn);
				out.m_layers[l].get_weights(wa)[w] = r.m_layers[l].get_weights(wa)[w];
				out.m_layers[l].get_delta_weights(wa)[w] = r.m_layers[l].get_delta_weights(wa)[w];
			}
		}
	}
	return out;
}
