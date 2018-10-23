#include <u.h>
#include <libc.h>
#include <ann.h>

double
fmax(double a, double b)
{
	if (a > b)
		return a;
	return b;
}

Ann*
adamaxinit(Ann *ann)
{
	int i;
	Adam *I = calloc(1, sizeof(Adam));

	I->rate = 0.002;
	I->beta1 = 0.9;
	I->beta2 = 0.999;
	I->epsilon = 10e-8;
	I->timestep = 0;
	I->first = calloc(ann->n-1, sizeof(Weights*));
	I->second = calloc(ann->n-1, sizeof(Weights*));

	for (i = 0; i < (ann->n-1); i++) {
		I->first[i] = weightscreate(ann->layers[i]->n, ann->layers[i+1]->n, 0);
		I->second[i] = weightscreate(ann->layers[i]->n, ann->layers[i+1]->n, 0);
	}

	ann->internal = I;

	return ann;
}

double
anntrain_adamax(Ann *ann, double *inputs, double *outputs)
{
	double *error = annrun(ann, inputs);
	double ret = 0.0;
	int noutputs = ann->layers[ann->n-1]->n;
	double acc, sum, m, v;
	int o, i, w, n;
	Neuron *O, *I;
	Weights *W, *D, *D2, *M, *V;
	Adam *annI;

	if (ann->internal == 0)
		adamaxinit(ann);
	annI = ann->internal;
	annI->timestep++;

	for (o = 0; o < noutputs; o++) {
		// error = outputs[o] - result
		error[o] -= outputs[o];
		error[o] = -error[o];
		ret += pow(error[o], 2.0) * 0.5;
	}
	D = ann->deltas[ann->n-2];
	weightsinitdoubles(D, error);
	for (i = 0; i < (ann->n-2); i++) {
		D = ann->deltas[i];
		weightsinitdouble(D, 1.0);
	}

	// backpropagate MSE
	D2 = ann->deltas[ann->n-2];
	for (w = ann->n-2; w >= 0; w--) {
		D = ann->deltas[w];
		M = annI->first[w];
		V = annI->second[w];

		for (o = 0; o < ann->layers[w+1]->n; o++) {
			O = ann->layers[w+1]->neurons[o];
			acc = O->gradient(O) * O->steepness;
			sum = 1.0;
			if (D2 != D) {
				W = ann->weights[w+1];
				sum = 0.0;
				for (n = 0; n < D2->outputs; n++)
					sum += D2->values[o][n] * W->values[o][n];
			}
			for (i = 0; i <= ann->layers[w]->n; i++) {
				I = ann->layers[w]->neurons[i];
			 	D->values[i][o] *= acc * sum;
				M->values[i][o] *= annI->beta1;
				M->values[i][o] += (1.0 - annI->beta1) * D->values[i][o] * I->value;
				V->values[i][o] = fmax(V->values[i][o] * annI->beta2, fabs(D->values[i][o] * I->value));
			}
		}

		D2 = D;
	}

	// update weights
	for (w = 0; w < ann->n-1; w++) {
		W = ann->weights[w];
		M = annI->first[w];
		V = annI->second[w];

		for (i = 0; i <= W->inputs; i++) {
			for (o = 0; o < W->outputs; o++) {
				m = M->values[i][o];
				v = V->values[i][o];
				W->values[i][o] += (annI->rate/(annI->timestep < 100? 1.0 - pow(annI->beta1, annI->timestep): 1.0)) * (m/v);
			}
		}
	}

	free(error);
	return ret;
}
