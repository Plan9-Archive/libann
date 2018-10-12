#include <u.h>
#include <libc.h>
#include <ann.h>

double*
annrun(Ann *ann, double *input)
{
	int l, i, o;
	int outputs = ann->layers[ann->n - 1]->n;
	double *ret = calloc(outputs, sizeof(double));
	Neuron *O;

	for (i = 0; i < ann->layers[0]->n; i++)
		ann->layers[0]->neurons[i]->value = input[i];

	for (l = 1; l < ann->n; l++) {
		for (o = 0; o < ann->layers[l]->n; o++) {
			O = ann->layers[l]->neurons[o];
			O->sum = 0;
			for (i = 0; i < ann->layers[l-1]->n; i++)
				O->sum += ann->layers[l-1]->neurons[i]->value * ann->weights[l-1]->values[i][o];
			O->value = O->activation(O->sum);
		}
	}

	for (o = 0; o < outputs; o++)
		ret[o] = ann->layers[ann->n - 1]->neurons[o]->value;

	return ret;
}
