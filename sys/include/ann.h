#pragma lib "libann.a"
#pragma src "/sys/src/libann"

typedef struct Ann Ann;
typedef struct Layer Layer;
typedef struct Neuron Neuron;
typedef struct Weights Weights;

struct Ann {
	int n;
	double rate;
	Layer **layers;
	Weights **weights;
	Weights **deltas;
};

struct Layer {
	int n;
	Neuron **neurons;
};

struct Neuron {
	double (*activation)(double input);
	double (*gradient)(double input);
	double steepness;
	double value;
	double sum;
};

struct Weights {
	int inputs;
	int outputs;
	double **values;
};

Ann *anncreate(int, ...);
Layer *layercreate(int, double(*)(double), double(*)(double));
Neuron *neuroninit(Neuron*, double (*)(double), double (*)(double), double);
Neuron *neuroncreate(double (*)(double), double (*)(double), double);
Weights *weightsinitrand(Weights*);
Weights *weightsinitdoubles(Weights*, double*);
Weights *weightscreate(int, int, int);
double *annrun(Ann*, double*);
void anntrain(Ann*, double*, double*);

