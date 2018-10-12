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
	double (*activation)(Neuron*);
	double (*gradient)(Neuron*);
	double steepness;
	double value;
	double sum;
};

struct Weights {
	int inputs;
	int outputs;
	double **values;
};

double activation_sigmoid(Neuron*);
double gradient_sigmoid(Neuron*);
Ann *anncreate(int, ...);
Layer *layercreate(int, double(*)(Neuron*), double(*)(Neuron*));
Neuron *neuroninit(Neuron*, double (*)(Neuron*), double (*)(Neuron*), double);
Neuron *neuroncreate(double (*)(Neuron*), double (*)(Neuron*), double);
Weights *weightsinitrand(Weights*);
Weights *weightsinitdouble(Weights*, double);
Weights *weightsinitdoubles(Weights*, double*);
Weights *weightscreate(int, int, int);
double *annrun(Ann*, double*);
void anntrain(Ann*, double*, double*);
