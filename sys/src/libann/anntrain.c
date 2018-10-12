#include <u.h>
#include <libc.h>
#include <ann.h>

void
anntrain(Ann *ann, double *inputs, double *outputs)
{
	double *error = annrun(ann, inputs);
	int noutputs = ann->layers[ann->n-1]->n;
	double acc, sum;
	int o, i, w, n;
	Neuron *O, *I;
	Weights *W, *D, *D2;

	for (o = 0; o < noutputs; o++) {
		// error = outputs[o] - result
		error[o] -= outputs[o];
		error[o] = -error[o];
	}
	D = ann->deltas[ann->n-2];
	weightsinitdoubles(D, error);

	// backpropagate MSE
	D2 = ann->deltas[ann->n-2];
	for (w = ann->n-2; w >= 0; w--) {
		D = ann->deltas[w];

		for (o = 0; o < ann->layers[w+1]->n; o++) {
			O = ann->layers[w+1]->neurons[o];
			acc = O->gradient(O) * O->steepness;
			sum = 1.0;
			if (D2 != D) {
				W = ann->weights[w + 1];
				sum = 0.0;
				for (n = 0; n < D2->outputs; n++)
					sum += D2->values[o][n] * W->values[o][n];
			}
			for (i = 0; i <= ann->layers[w]->n; i++) {
			 	D->values[i][o] *= acc * sum;
			}
		}

		D2 = D;
	}

	// update weights
	for (w = 0; w < ann->n-1; w++) {
		W = ann->weights[w];
		D = ann->deltas[w];

		for (i = 0; i <= W->inputs; i++) {
			I = ann->layers[w]->neurons[i];
			for (o = 0; o < W->outputs; o++) {
				W->values[i][o] += D->values[i][o] * ann->rate * I->value;
			}
		}
	}

	free(error);
}
