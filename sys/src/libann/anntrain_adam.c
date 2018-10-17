#include <u.h>
#include <libc.h>
#include <ann.h>

Ann*
adaminit(Ann *ann)
{
	int i;
	Adam *I = calloc(1, sizeof(Adam));

	I->rate = 0.001;
	I->beta1 = 0.75;
	I->beta2 = 0.9;
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
anntrain_adam(Ann *ann, double *inputs, double *outputs)
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
		adaminit(ann);
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
			 	D->values[i][o] *= acc * sum;
				M->values[i][o] *= annI->beta1;
				M->values[i][o] += (1.0 - annI->beta1) * D->values[i][o];
				V->values[i][o] *= annI->beta2;
				V->values[i][o] += (1.0 - annI->beta2) * D->values[i][o] * D->values[i][o];
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
			I = ann->layers[w]->neurons[i];
			for (o = 0; o < W->outputs; o++) {
				m = M->values[i][o] / (annI->timestep < 1000? 1.0 - pow(annI->beta1, annI->timestep): 1.0);
				v = V->values[i][o] / (annI->timestep < 1000? 1.0 - pow(annI->beta2, annI->timestep): 1.0);
				W->values[i][o] += (m / (sqrt(v) + annI->epsilon)) * annI->rate * I->value;
			}
		}
	}

	free(error);
	return ret;
}
