# net
## Copyright
Public domain, 2023

github.com/SirJonthe

## About
`net` is a minimalist library for C++11 that implements a fully connected neural network.

## Design
`net` is designed to be as minimalist and easy to use as possible while still achieving the goal of being a handy tool. More 

## Usage
`net` provides only a few contact points for the programmer to use. Create a `net` object with a specific topography, and either use `train` to train it, or `feed_forward` to predict an outcome based on the inputs. The results can be viewed in `get_output_layer`.

## Building
No special adjustments need to be made to build `net` except enabling C++11 compatibility or above. Simply include the relevant headers in your code and make sure the headers and source files are available in your compiler search paths. Using `g++` as an example, building is no harder than:

```
g++ -std=c++11 code.cpp net/net.cpp
```

...where `code.cpp` is an example source file containing the user-defined commands as well as the entry point for the program.

## Examples
### Creating a neural network:
Create a network from a fixed-size array via constructor:
```
#include "net/net.h"

int main()
{
	int TOPGRAPHY[] = { 1, 2, 3 };
	cc0::net n(TOPOGRAPHY);
	return 0;
}
```

Create a network from a variable-size array via constructor:
```
#include "net/net.h"

int main()
{
	int *TOPGRAPHY = new int[3];
	TOPOGRAPHY[0] = 1;
	TOPOGRAPHY[1] = 2;
	TOPOGRAPHY[2] = 3;
	cc0::net n(TOPOGRAPHY, 3); // 3 corresponds to the number of elements in the variable-size TOPOGRAPHY array.
	return 0;
}
```

Create a network from a fixed-size array via constructor:
```
#include "net/net.h"

int main()
{
	int TOPGRAPHY[] = { 1, 2, 3 };
	cc0::net n;
	n.create(TOPOGRAPHY);
	return 0;
}
```

Create a network from a variable-size array via constructor:
```
#include "net/net.h"

int main()
{
	int *TOPGRAPHY = new int[3];
	TOPOGRAPHY[0] = 1;
	TOPOGRAPHY[1] = 2;
	TOPOGRAPHY[2] = 3;
	cc0::net n;
	n.create(TOPOGRAPHY, 3); // 3 corresponds to the number of elements in the variable-size TOPOGRAPHY array.
	return 0;
}
```
The numbers in the topography corresponds to the number of neurons that should be allocated per layer in the network. The first index in the topography is the input layer neuron count, and the last index is the output layer neuron count.

### Creating a neural network using custom randomization:
The initial state of the network is randomized. It uses a standard method of randomization, but the programmer may provide their own randomization method to initialization. The below examples assumes the programmer has implemented a function called `custom_rand` which takes no parameters and returns a floating point value in the 0-1 range.

Create a network from a fixed-size array via constructor:
```
#include "net/net.h"

int main()
{
	int TOPGRAPHY[] = { 1, 2, 3 };
	cc0::net n(TOPOGRAPHY, custom_rand);
	return 0;
}
```

Create a network from a variable-size array via constructor:
```
#include "net/net.h"

int main()
{
	int *TOPGRAPHY = new int[3];
	TOPOGRAPHY[0] = 1;
	TOPOGRAPHY[1] = 2;
	TOPOGRAPHY[2] = 3;
	cc0::net n(TOPOGRAPHY, 3, custom_rand);
	return 0;
}
```

Create a network from a fixed-size array via constructor:
```
#include "net/net.h"

int main()
{
	int TOPGRAPHY[] = { 1, 2, 3 };
	cc0::net n;
	n.create(TOPOGRAPHY, custom_rand);
	return 0;
}
```

Create a network from a variable-size array via constructor:
```
#include "net/net.h"

int main()
{
	int *TOPGRAPHY = new int[3];
	TOPOGRAPHY[0] = 1;
	TOPOGRAPHY[1] = 2;
	TOPOGRAPHY[2] = 3;
	cc0::net n;
	n.create(TOPOGRAPHY, 3, custom_rand);
	return 0;
}
```

### Train the network:
Training the network is done by providing the `train` function with both an input value array and an expected output value array:
```
#include "net/net.h"

int main()
{
	int TOPGRAPHY[] = { 2, 3, 4 };
	cc0::net n(TOPOGRAPHY);

	const float IN[2] = { 1.0f, 0.5f };
	const float OUT[4] = { 0.5f, 0.5f, 1.0f, 0.75f };

	n.train(IN, OUT);

	return 0;
}
```
Note that the input array size is assumed to correspond to the specified number of neurons in the first index of the topography. Similarly, the expected output array size is assumed to correspond to the specified number of neurons in the last index of the topography. If not the program may crash.

### Using the network to classify inputs:
Using the network for classification is done by providing the `feed_forward` function with an input value array and reading from the output layer using `get_output_layer`:
```
#include "net/net.h"
#include <iostream>

int main()
{
	int TOPGRAPHY[] = { 2, 3, 4 };
	cc0::net n(TOPOGRAPHY);

	const float IN[2] = { 1.0f, 0.5f };
	n.feed_forward(IN, OUT);

	for (uint64_t i = 0; i < n.get_output_layer().get_neuron_count(); ++i) {
		std::cout << n.get_output_layer().get_neurons()[i] << std::endl;
	}

	return 0;
}
```
Note that the input array size is assumed to correspond to the specified number of neurons in the first index of the topography.

### Deallocate memory:
The network's memory is automatically cleaned up when it falls out of scope. The programmer may, however, also choose to free memory manually:
```
#include "net/net.h"

int main()
{
	int TOPGRAPHY[] = {1, 2, 3};
	cc0::net n(TOPOGRAPHY); // Allocate the network.
	n.destroy(); // Deallocate the network.
	return 0;
}
```

## Future work
There are currently a number of improvements that need to be done with regards to both performance and memory consumption. First, the network can benefit from CPU vector extensions (I do not want to introduce GPU calculations for this library) and perhaps multi-threading. Secondly, there could be a possibility where many of the hyper-parameters are not stored in memory, thus drastically reducing the memory consumption of the model.

There are also a few missing functionalities that should be added at some point; Unsupervised learning for situations where the user does not necessarily know the solution or the training data is too large to classify manually, and non-fully connected networks which may perform better for certain tasks.

`net` is also mostly an untested library. While it seems to be doing what it should, it is difficult to verify how accurate it is without testing it inside a practical application. I should eventually test it out on a MNIST dataset.
