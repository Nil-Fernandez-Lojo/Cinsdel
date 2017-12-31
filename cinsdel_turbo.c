/* *****************************************************************************************
   	Copyright Jossy Sayir and Nil Fernandez Lojo

   	The following code performs simulations on the insertion deletion substitution (insdel)
   	channel. The aim is to perform serial turbo coding using convolutional codes and LDPC
   	codes.
   	Different modes:
		1: measure_I_one_point : measure the (iid) capacity of an insertion deletion 
		channel (insdel).
			input: 1 Ni md pi pd ps gamma seed

		2: oneshot_viterbi: simulate the correction of 1 message coded with a convolutional 
		code (cc) through the insdel channel using the Viterbi algorithm
			input:  p1 p2 rec Ni md pi pd ps

		3: oneshot_BCJR: simulate the correction of 1 message coded with a 
		convolutional code (cc) through the insdel channel using the BCJR algorithm
			input: 3 p1 p2 rec Ni md pi pd ps

		4: oneshot_LDPC: simulate the correction of 1 message coded with LDPC through the binary 
		symmetric channel (BSC)
			input: 4 ps file_name_matrixes

		5: oneshot_markers: simulate the correction of 1 message coded with a convolutional 
		code (cc) through the insdel channel using Marker codes.
			input: 5 d Ni md pi pd ps

		6: oneshot_bcjr_LDPC: simulate the correction of 1 message coded with serial 
		concatenation turbo codingusing LDPC for outer coding and cc for inner coding through 
		the insdel channel.
			input: 6 p1 p2 rec md pi pd ps file_name_matrixes

		7: oneshot_Marker_LDPC: simulate the correction of 1 message coded with iterative decoding
		using LDPC for outer coding and Marker for inner coding through the insdel channel.
			input: 7 d md pi pd ps file_name_matrixes

		8: batch_viterbi: Does the same simulation as in mode 2 (Viterbi) for several iterations
		(at least MIN_BLOCKS iterations and stops when it has done MAX_BLOCKS iterations or when 
		it has been running for MAX_SIM_TIME seconds). The program will open the outfile and 
		write 1 line which is described by output below
			input: 8 p1 p2 rec Ni md pi pd ps outfile seed
			output: seed p1 p2 rec Ni md pi pd ps block err dblock totpasses

		9: batch_bcjr: Does the same simulation as in mode 3 (BCJR) for several iterations (at 
		least MIN_BLOCKS iterations and stops when it has done MAX_BLOCKS iterations or when it 
		has been running for MAX_SIM_TIME seconds). The program will open the outfile and write
		1 line which is described by output below
			input: 9 p1 p2 rec Ni md pi pd ps outfile seed
			output: seed p1 p2 rec Ni md pi pd ps block err dblock totpasses

		10: batch_markers: Does the same simulation as in mode 4 (Marker codes) for several 
		iterations (at least MIN_BLOCKS iterations and stops when it has done MAX_BLOCKS iterations 
		or when it has been running for MAX_SIM_TIME seconds). The program will open the outfile 
		and write 1 line which is described by output below
			input: 10 d Ni md pi pd ps outfile seed
			output: seed d Ni md pi pd ps block err dblock totpasses

		11: batch_LDPC: Does the same simulation as in mode 4 (LDPC) for several iterations (at
		least MIN_BLOCKS iterations and stops when it has done MAX_BLOCKS iterations or when it 
		has been running for MAX_SIM_TIME seconds). The program will open the outfile and write 
		1 line which is described by output below
			input: 11 ps file_name_matrixes outfile seed
			output: seed Ni ps block err bit_err num_it_LDPC

		12: Batch turbo BCJR, not done yet

		13: Batch iterative Marker LDPC, not done yet

		14:batch_bcjr_EXIT:  Computes one point of the EXIT chart (assuming Gaussian dirstribution
		of the a priori LLRs) and writes the result in the outfile. The outfile_LLR (optional)
		writes all the LLRs. 
			input: 14 p1 p2 rec Ni md pi pd ps outfile seed I_in outfile_LLR
			output: seed p1 p2 Ni maxdelta pi pd ps I_in I_out block
			output_LLR: block, u[k], LLR[k]

		15: batch_Marker_EXIT:  Computes one point of the EXIT chart (assuming Gaussian dirstribution
		of the a priori LLRs) and writes the result in the outfile.
			input: 15 d Ni md pi pd ps outfile seed I_in
			output: seed d Ni maxdelta pi pd ps I_in I_out block

		16: batch_markers_LDPC: Does the same simulation as in mode 7 (Marker_LDPC) for several iterations (at
		least MIN_BLOCKS iterations and stops when it has done MAX_BLOCKS iterations or when it 
		has been running for MAX_SIM_TIME seconds). The program will open the outfile and write 
		1 line which is described by output below
			input: 16 d maxdelta pi pd ps outfile file_name_matrixes seed
			output: d, maxdelta, pi, pd, ps,block, err
		
	Definition of the input variables
		Ni: is the number of bits of the input message
		md: is the width of the band in the 3D Trellis i.e only consider output messages of
			length = rate * #input_bits ± md
		pi: is the probability of insertion
		pd: is the probability of deletion
		ps: is the probability of substitution
		seed: is the RNG seed
		p1: first polynomial decribing the cc (in octal). Descirption of the meaning of the
			polynomial in the function conv_convert
		p2: second polynomial (in octal). If it is 0, the cc is has a rate 1, otherwise it is
			has a rate 1/2.
		rec: if rec is 0, the cc is non recursive, if it is 1, it is recursive
		file_name_matrixes: is the name of the file constaining the generator and parity check
			matrixes. It uses the format from the PEG implementation of Xiaoyu Hu (see peg in
			the Github) using output mode 2.
		outfile: name of the file when the program will write its output
		I_in: the mutual information at the input
		d: number of data points between markers
		gamma: transition probabilty in Markov source

	Definition of extra output variables
		block: number of iterations of the batch simulation
		err : number of iterations where the message was incorrectly decoded or where the program
			couldn't decode the message
		dblock : number of iterations where the program managed to decode the message (correctly
			or incorrectly)
		totpasses : total number of passes in the trellis
		num_it_LDPC: total number of LDPC iterations
		I_out: the mutual information at the output
   ***************************************************************************************** */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define LABELLENGTH 82
#define MSPT 5 // maximum number of symbols allowable per transition
#define MAXT 12 // maximum number of transitions from or to state
#define EPSILON 1e-5 // precision for calculations 
#define MAX_TRANSMIT_LENGTH 20000
#define MIN_ERRORS 10000
#define MIN_BLOCKS 1
#define MAX_BLOCKS 20000
#define MAX_SIM_TIME 36000
#define MAX_LDPC_IT 5	
#define MAX_ROWS 20000 //for the parity check matrix
#define MAX_OUT_IT_TURBO 50
#define MAX_PACKETS_EXIT_SIM 1000
#define ITERATIONS_MEASURE_CAPACITY 100
#define MAX_L 500


#define SEED 324374832U  // "seed" of the random number generator.
// this can be any weird number, don't use 0 or something too simple
// Best leave this fixed to get repeatability of all experiments
// HOWEVER: be aware that two runs of the programme with the same
// parameters will generate EXACTLY THE SAME SEQUENCES! If you want
// different sequences, recompile with a different seed value or
// alter the programme to take a seed as an external parameter.

/* *********************************************************************************************
 * Function definitions and globals                                                            *
 ********************************************************************************************* */
void init_genrand(unsigned long s);
unsigned long genrand_int32(void);
double genrand_real1(void);
double genrand_exp1(void);
double genrand_norm(void);
int log2int(unsigned long x);

char randmarkers[] = "1101100111011010011110111110101000011010001100011101100010101011111000101010001001111011010011101000010101011100010111000101110001010000111011010000000011000100100000111000100011101010100110110000111110110111110000100000010011000010110000010010110100111001100101110001010101111010011011111100100011100100101110111110010000110010110001000000110100110101111100100111000101100000100100101110101110100000001011100011011110011000000101111101011000110110101000010100010001010101000111011111010010011010110111100011011111110000000111110010111001110010010010101100000010101011001101011011111000111010001000001111111101111010011111010111111111001010110100000000010110100011001100100001101110111111000010000101110000101011110001100001000110101110100010000010000010000011100111010010011111101100101100101110001110100101111011100011100010010100100010000101101101010010100010010011000001110100000000001110001110011000010101000110101110000011000000111001111010001001111011101101010000011101110010011110010111111001";

char markers[] = "001110";

/* *****************************************************************************************
   TRANSDUCER DATA STRUCTURES DEFINITIONS
   ***************************************************************************************** */

// transducer state with label, forward links and backward links

typedef struct {
	char label[LABELLENGTH];

	int nf; // number of forward transitions
	unsigned long f[MAXT]; // forward transitions (index pointers)

	int nb; // number of backward transitions
	unsigned long b[MAXT]; // backward transitions (index pointers)

} state;

// transition between transducer states, origin state, destination state, weight/probability of transition,
// input tape symbols and output tape symbols associated with transition

typedef struct {
	unsigned long os; // origin state (index pointer)
	unsigned long ds; // destination state (index pointer)

	double p; // weight / probability of the branch
	//int p; // weight / index to probability of the branch
	// NOTE: switched from storing actual probabilities on a branch to storing an index
	// that can be used to look up a probability in a separate array. This way, we can
	// maintain a transducer structure and switch the separate array to adapt to new
	// channel parameters, and there is no need to reconstruct a transducer from scratch

	int ni; // number of input symbols
	char i[MSPT]; // input symbols

	int no; // number of output symbols
	char o[MSPT]; // output symbols

} transition;

// transducer: states and transitions with their numbers

typedef struct {
	state *s; // array of transducer states
	unsigned long ns; // number of states

	transition *t; // array of transitions
	unsigned long nt; // number of transitions

	// dimensions
} transducer;

// a trellis is a transducer built from an underlying transducer where the number of symbols
// in the input and output tapes are added to the states of the underlying transducer.
// These numbers are computed implicitly through addressing in our implementiation
// i.e. the state at position trelsind(trel,ni,no,s) is equivalent to the underlying transducer
// state s with ni symbols on the input tape and no symbols on the output tape.

typedef struct {
	transducer *td; // trellis transducer (not underlying transducer)
	unsigned long nsund; // number of states of the underlying transducer
	unsigned long ntund; // number of transitions of the underlying transducer
	int nni; // maximum number of input symbols
	//  int nno; // maximum number output symbols
	// the following implements a "banded" trellis
	int *av_no; // average number of output symbols for each input length
	int maxdelta; // maximum deviation from average in trellis

	// following external arrays extend the states or transitions in the trellis and
	// are used by various decoders. These variables could have been inserted into the
	// state and transition structures but it's clearer to keep these as minimal as
	// necessary to store the transducer structure and to keep any decoding metrics
	// in separate arrays
	double *tm; // transition metrics for decoder
	double *fm; // forward metrics for decoder
	double *bm; // backward metrics for decoder
	unsigned long *bt; // backward pointers for Viterbi
} trellis;

/* *****************************************************************************************
   LDPC STRUCTURES
   ***************************************************************************************** */
typedef struct {
	int N; // number of columns
	int M; // number of rows
	int *ones_per_row; // number of ones per row but for the case of the generator matrix,
	// it is the number ones per column
	int *rows; // indexes of ones in the rows (stored in a 1d array) but for the case
	// of the generator matrix, it is the indexes of ones in the columns
} sparse_matrix;


typedef struct {
	int N; // number of columns
	int M; // number of rows
	int number_of_ones;
	int *col_connect;
	int *row_connect;
	int *interleaver;
} Tanner_graph;

/* *****************************************************************************************
   TRANSDUCER I/O FUNCTIONS
   ***************************************************************************************** */

// ffprint_transducer prints a transducer to a stream
// note that empty input and output alphabets are represented with the special character '.'

void ffprint_transducer(transducer *td, FILE *f)
{
	unsigned long j, k;
	char i[MSPT], o[MSPT];
	for (j = 0 ; j < td->nt ; j++) {
		//if (td->t[j].ni == 0 && td->t[j].no == 0)
		//continue;
		if (td->t[j].ni == 0)
			strcpy(i, ".");
		else
			strcpy(i, td->t[j].i);
		if (td->t[j].no == 0)
			strcpy(o, ".");
		else
			strcpy(o, td->t[j].o);
		fprintf(f, "%s %s %s %s %f\n", td->s[td->t[j].os].label, td->s[td->t[j].ds].label, i, o, td->t[j].p);
	}
}

// print transducer to stdout

void print_transducer(transducer *t)
{
	ffprint_transducer(t, stdout);
}

// print transducer to stdout followed by carriage return

void println_transducer(transducer *t)
{
	ffprint_transducer(t, stdout);
	printf("\n");
}

// print transducer to file specified by filename

void fprint_transducer(transducer *t, char *filename)
{
	FILE *f;
	assert ((f = fopen(filename, "w")) != NULL);
	ffprint_transducer(t, f);
	fclose(f);
}
/*
// find_or_create_state INTERNAL function used by transducer loading function to search for a state with a given label
// and create it if not yet available

unsigned long find_or_create_state(transducer *td, char *label)
{
	unsigned long k;

	for (k = 0UL ; k < td->ns ; k++)
		if (strcmp(td->s[k].label, label) == 0)
			break;
	if (k == td->ns) { // new state
		strcpy(td->s[k].label, label);
		td->s[k].nf = 0;
		td->s[k].nb = 0;
		td->ns++;
	}
	return (k);
}

// ffload_transducer loads transducer from stream
// SYNTAX: stream must provide a list of transducer transitions
// each entry in file must contain 5 entries separated by spaces:
//  origin_state destination_state input_symbols output_symbols probability_of_transition
// the origin_state and destination_state can be described by any string (no spaces or '#' allowed)
// the input symbols and output symbols can be over any alphabet (no spaces, '#' or '.') and written as strings
// a single '.' denotes an empty string
// the probabilities will be read as double-precision floats, and must not necessarily sum to 1. If they don't sum to 1
// they will be interpreted as "weights" and re-normalised to sum to 1. Any negative weight will be ignored and set to 0.0
// COMMENTS can be added at will after the 5 entries in a line. Any line that does not begin with 5 entries, the 5th of
// which can be parsed as a double-precision float will be ignored and hence comments can be written in free form as long
// as the 5th word is not a number. Any text or entries after a character '#' will be ignored in any case, so a better
// way to add comments is to precede a comment line with '#'.

transducer* ffload_transducer(FILE *f)
{
	char ln[500], os[LABELLENGTH], ds[LABELLENGTH], i[MSPT], o[MSPT], *c;
	unsigned long j, k;
	transducer *td;
	state *s;
	int p;


	// allocate space for a new transducer and count transitions and allocate memory space for them
	assert((td = malloc(sizeof(transducer))) != NULL);
	// count transitions
	for (td->nt = 0UL ; fgets(ln, 500, f) == ln ; ) {
		if ((c = strchr(ln, '#')) != NULL) // cut at comment
			*c = '\0';
		if (sscanf(ln, "%s %s %s %s %d", os, ds, i, o, &p) == 5)
			td->nt++;
	}
	rewind(f);
	assert((td->t = malloc(td->nt * sizeof(transition))) != NULL);

	// pre-allocate double the number of transitions for states (this is an upper bound on the number
	// of states and this number will be cut back once all actual states have been parsed and counted)
	assert((td->s = malloc(2 * td->nt * sizeof(state))) != NULL);
	td->ns = 0UL; // current total state counter

	// enter all transitions into transducer and connect them, creating states as required
	for (j = 0 ; j < td->nt ; j++) {
		// get next data-carrying line
		while (fgets(ln, 500, f) == ln) {
			if ((c = strchr(ln, '#')) != NULL) // cut at comment
				*c = '\0';
			if (sscanf(ln, "%s %s %s %s %d", os, ds, i, o, &p) == 5)
				break; // found a data-carrying line
		}
		// connect transition to origin state
		k = find_or_create_state(td, os);
		td->t[j].os = k;
		s = &(td->s[k]);
		s->f[s->nf] = j;
		s->nf++;
		// connect transition to destination state
		k = find_or_create_state(td, ds);
		td->t[j].ds = k;
		s = &(td->s[k]);
		s->b[s->nb] = j;
		s->nb++;
		// populate input and output symbols for transition
		if (strcmp(".", i) == 0) { // empty string
			td->t[j].ni = 0;
			td->t[j].i[0] = '\0';
		}
		else {
			td->t[j].ni = strlen(i);
			assert(td->t[j].ni < MSPT);
			strcpy(td->t[j].i, i);
		}
		if (strcmp(".", o) == 0) { // empty string
			td->t[j].no = 0;
			td->t[j].o[0] = '\0';
		}
		else {
			td->t[j].no = strlen(o);
			assert(td->t[j].no < MSPT);
			strcpy(td->t[j].o, o);
		}
		// assign weight / probability to transition
		td->t[j].p = p;
	}
	// we had allocated an upper bound for td->s, now we can reduce to actual size
	assert((td->s = realloc(td->s, td->ns * sizeof(state))) != NULL);

	return (td);
}


// load transducer from standard input

transducer *load_transducer()
{
	return (ffload_transducer(stdin));
}

// load a transducer from a named file

transducer *fload_transducer(char *filename)
{
	FILE *f;
	transducer *t;
	assert((f = fopen(filename, "r")) != NULL);
	t = ffload_transducer(f);
	fclose(f);
	return (t);
}*/

// free all memory allocated for a transducer

void free_transducer(transducer *td)
{
	free(td->t);
	free(td->s);
	free(td);
}


/* *****************************************************************************************
   TRANSDUCER MANIPULATION FUNCTIONS
   ***************************************************************************************** */

/* transform a transducer into a new transducer that has one output
   symbol per transition, introducing extra states where necessary */
// "timeout" read here  as "time-out", i.e., time the transducer to match one symbol per output

transducer *timeout_transducer(transducer *tdi)
{
	transducer *tdo;
	unsigned long j, k, ns, nt;

	assert((tdo = malloc(sizeof(transducer))) != NULL);
	// count the number of transitions and states of new transducer
	for (j = 0UL, tdo->nt = 0UL, tdo->ns = tdi->ns ; j < tdi->nt ; j++)
		if (tdi->t[j].no == 0)
			tdo->nt++; // a zero output transition remains in new transducer
		else {
			tdo->nt += tdi->t[j].no; // transitions for every output symbol
			tdo->ns += (tdi->t[j].no - 1); // no - 1 intermediate states will be necessary
		}
	// allocate memory for new transducer
	assert((tdo->t = malloc(tdo->nt * sizeof(transition))) != NULL);
	assert((tdo->s = malloc(tdo->ns * sizeof(state))) != NULL);

	// first copy old transducer to new transducer
	memcpy(tdo->t, tdi->t, tdi->nt * sizeof(transition));
	memcpy(tdo->s, tdi->s, tdi->ns * sizeof(state));

	// add intermediate states
	ns = tdi->ns; // index of first available new state for assignment
	nt = tdi->nt; // index of first available new transition for assignment
	for (j = 0 ; j < tdi->nt ; j++) {
		// if this transition has 0 or 1 outputs, no new intermediate states are required
		if (tdo->t[j].no <= 1)
			continue;
		// current transition now points to first new intermediate state with 1 output symbol
		tdo->t[j].ds = ns;
		tdo->t[j].no = 1;
		sprintf(tdo->t[j].o, "%c", tdi->t[j].o[0]);
		// create new intemediate states and connect them
		for (k = 1 ; k < tdi->t[j].no ; k++) {
			// create new state
			// new state has label derived from old origin and destination states, and 1 in and 1 out transitions
			sprintf(tdo->s[ns].label, "%s_%s_%.2lu", tdi->s[tdi->t[j].os].label, tdi->s[tdi->t[j].ds].label, k);
			tdo->s[ns].nf = 1;
			tdo->s[ns].nb = 1;
			if (k == 1) // connect back to original transition
				tdo->s[ns].b[0] = j;
			else { // connect back to previously generated new transition
				tdo->s[ns].b[0] = nt - 1;
			}
			// create new forward transition from new state
			tdo->s[ns].f[0] = nt;
			tdo->t[nt].os = ns;
			if (k == tdi->t[j].no - 1) // last transition must point to old destination state
				tdo->t[nt].ds = tdi->t[j].ds;
			else // point to next new state that will be creatd
				tdo->t[nt].ds = ns + 1;
			
			if (k == 1) tdo->t[nt].p = tdi->t[j].p; //the probability of the transition is the same for the first output
			else tdo->t[nt].p = 0; //the probability of the transition is 1 from an intermediate state to a real one

			tdo->t[nt].ni = 0;
			tdo->t[nt].i[0] = '\0';
			tdo->t[nt].no = 1;
			sprintf(tdo->t[nt].o, "%c", tdi->t[j].o[k]);
			// increment state and transition pointers
			ns++;
			nt++;
		}
		// find the backward pointer of the destination state that corresponded to the transition
		// and reconnect to the new transition from the last intermediate state
		for (k = 0 ; k < tdi->s[tdi->t[j].ds].nb ; k++)
			if (tdo->s[tdi->t[j].ds].b[k] == j)
				break;
		assert(k != tdi->s[tdi->t[j].ds].nb); // should always find a j
		tdo->s[tdi->t[j].ds].b[k] = nt - 1;

	}
	return (tdo);
}

/* the next function is used internally by the transducer composition function... */

void composite_transition(transducer *tdl, transducer *tdr, transducer *tdo, unsigned long tl, unsigned long tr, unsigned long to, int silentflag)
{
	// silentflag indicates if we are dealing is a silent transition in the left transducer
	// output or right transducer input, in which case the transition pointer in the other
	// transducer is interpreted as a state pointer (the "transition" in the other transducer
	// is a conceptualised silent in silent out transition going from the state to itself)
	// silentflag 0 means no silent transition, 1 means a silent transition on the right
	// (tl is a state pointer) and -1 means a silent transition on the left (tr is a state pointer)

	unsigned long oind, dind;


	// state index of origin and destinationstate in new transducer
	switch (silentflag) {
	case 1:
		oind = tl * tdr->ns + tdr->t[tr].os;
		dind = tl * tdr->ns + tdr->t[tr].ds;
		tdo->t[to].ni = 0;
		tdo->t[to].no = tdr->t[tr].no;
		memcpy(tdo->t[to].o, tdr->t[tr].o, (tdo->t[to].no+1) * sizeof(char));
		//tdo->t[to].p = tdr->t[tr].p;
		break;
	case -1:
		oind = tdl->t[tl].os * tdr->ns + tr;
		dind = tdl->t[tl].ds * tdr->ns + tr;
		tdo->t[to].ni = tdl->t[tl].ni;
		tdo->t[to].no = 0;
		memcpy(tdo->t[to].i, tdl->t[tl].i, (tdo->t[to].ni+1) * sizeof(char));
		//tdo->t[to].p = tdl->t[tl].p;
		break;
	case 0:
	default:
		oind = tdl->t[tl].os * tdr->ns + tdr->t[tr].os;
		dind = tdl->t[tl].ds * tdr->ns + tdr->t[tr].ds;
		tdo->t[to].ni = tdl->t[tl].ni;
		tdo->t[to].no = tdr->t[tr].no;
		memcpy(tdo->t[to].i, tdl->t[tl].i, (tdo->t[to].ni+1) * sizeof(char));
		memcpy(tdo->t[to].o, tdr->t[tr].o, (tdo->t[to].no+1) * sizeof(char));


		// here we multiply the weights/probabilities... Having now switched to an indexed
		// approach to storing a weight for transitions, we still maintain the multiplication,
		// assuming that the external lookup table for weights will be organised as a 2-dimensonal
		// array so that any couple (i,j) of indices
		//tdo->t[to].p = tdl->t[tl].p * tdr->t[tr].p;
		// for the moment we will only deal with compositions of transducers where tdl has unweighted
		// (or equal weighted) transitions, so transitions are simply inherited from tdr. If we ever
		// change this, we would need to get the number of different types of transitions of each
		// transducer so p can be an index into a 2-dimensional array
		//tdo->t[to].p = tdr->t[tr].p;
	}
	tdo->t[to].p = tdl->t[tl].p + tdr->t[tr].p;

	// connect origin state
	tdo->t[to].os = oind;
	tdo->s[oind].f[tdo->s[oind].nf++] = to;
	// connect destination state
	tdo->t[to].ds = dind;
	tdo->s[dind].b[tdo->s[dind].nb++] = to;
}

// compose two transducers to obtain a new transducer where the output of the left transducer is fed
// into the input of the right transducer. The state space of the proposed transducer is the product
// of the two state spaces and so is the transition space, e.g., a transition of the new transducer
// corresponds to a combination of a transition of the left transducer (with its input) combined with
// a transition of the right transducer (with its output), going from the combined state of the two
// origin states to the combined state of the two destination states, and with probability the product
// of the two transition probabilities.

transducer *compose_transducers(transducer *tdl, transducer *tdr)
{
	unsigned long j, k, tind, nind, nzoutl, nzinr;
	transducer *tdo;

	// check that the transitions of tdl and transitions of rtd have
	// at most one output / input per branch, respectively.
	// count zero output transitions in the left transfducer and zero input
	// transitions in the right transducer
	for (j = 0UL, nzoutl = 0UL ; j < tdl->nt ; j++)
		switch (tdl->t[j].no) {
		case 0:
			nzoutl++;
		case 1:
			break;
		default:
			return (NULL);
		}
	for (j = 0UL , nzinr = 0UL ; j < tdr->nt ; j++)
		switch (tdr->t[j].ni) {
		case 0:
			nzinr++;
		case 1:
			break;
		default:
			return (NULL);
		}
	// allocate states and transitions for output transducer
	assert((tdo = malloc(sizeof(transducer))) != NULL);
	// the composed transducer will have transitions for every combination of matching alphabet
	// transitions of the left and right transducer (either one symbol from left to right, or
	// zero symbols from left to right) plus a transition for every left state for every zero
	// input transition on the right (this is equivalent to adding a zero input zero output transition
	// to every state in the left transition, which would not essentially modify the function of the
	// left transducer, then composing it and applying the rule above).
	// the memory allocated is overkill as it includes transitions for every possible combination of input/output
	// symbols but in fact only those with matching symbols will be used.
	// (this will later be corrected to the correct number of transitions used)
	tdo->nt = (tdl->nt - nzoutl) * (tdr->nt - nzinr) + nzoutl * nzinr + tdl->ns * nzinr + tdr->ns * nzoutl;;
	tdo->ns = tdl->ns * tdr->ns; // every combination of states gives a state
	assert((tdo->t = malloc(tdo->nt * sizeof(transition))) != NULL);
	assert((tdo->s = malloc(tdo->ns * sizeof(state))) != NULL);
	// label all new states with (lstate,rstate) labels and initialise to zero transitions
	for (j = 0 ; j < tdl->ns ; j++) {
		for (k = 0 ; k < tdr->ns ; k++) {
			nind = j * tdr->ns + k;
			sprintf(tdo->s[nind].label, "(%s,%s)", tdl->s[j].label, tdr->s[k].label);
			tdo->s[nind].nf = 0;
			tdo->s[nind].nb = 0;
		}
	}
	// label and connect all transitions in new transducer
	for (tind = 0UL , j = 0 ; j < tdl->nt ; j++)
		// if the left transducer has a silent output transition
		if (tdl->t[j].no == 0)  // add silent output transitions for every right state
			for (k = 0 ; k < tdr->ns ; k++)
				composite_transition(tdl, tdr, tdo, j, k, tind++, -1);

	for (j = 0 ; j < tdr->nt ; j++)
		if (tdr->t[j].ni == 0)
			for (k = 0 ; k < tdl->ns ; k++)
				composite_transition(tdl, tdr, tdo, k, j, tind++, +1);
	for (j = 0 ; j < tdl->nt ; j++)
		for (k = 0 ; k < tdr->nt ; k++)
			if ((tdl->t[j].no == 0 && tdr->t[k].ni == 0) ||
			        ((tdl->t[j].no == 1 && tdr->t[k].ni == 1) &&
			         (tdl->t[j].o[0] == tdr->t[k].i[0]))) {
				composite_transition(tdl, tdr, tdo, j, k, tind++, 0);
			}

	tdo->nt = tind;
	assert((tdo->t = realloc(tdo->t, tind * sizeof(transition))) != NULL);
	// perhaps re-normalise the probabilities to ensure that there is no
	// problem with the products ? (may implement this at a later stage.)
	return (tdo);
}

/* *****************************************************************************************
   SOURCE TRANSDUCERS
   ***************************************************************************************** */
transducer *Markov_source_transducer(double gamma)
{
	transducer *td;
	unsigned long j;
	assert((td = malloc(sizeof(transducer))) != NULL);

	if(gamma == 0.5) {
		td->ns = 1;
		assert((td->s = malloc(td->ns*sizeof(state))) != NULL);
		td->nt = 2;
		assert((td->t = malloc(td->nt * sizeof(transition))) != NULL);
		sprintf(td->s[0].label, "0");

		td->s[0].nf = 2;
		td->s[0].nb = 2;

		td->t[0].i[0] = '0';
		td->t[0].o[0] = '0';
		td->t[0].os = 0UL;
		td->t[0].ds = 0UL;
		td->s[0].f[0] = 0;
		td->s[0].b[0] = 0;

		td->t[1].i[0] = '1';
		td->t[1].o[0] = '1';
		td->t[1].os = 0UL;
		td->t[1].ds = 0UL;
		td->s[0].f[1] = 1;
		td->s[0].b[1] = 1;
		for (j = 0;j<2;j++) {
			td->t[j].p = log(0.5);
			td->t[j].ni = 1; 
			td->t[j].no = 1; 
			td->t[j].i[1] = '\0';
			td->t[j].o[1] = '\0';
		}
	}
	else {
		td->ns = 2;
		assert((td->s = malloc(td->ns*sizeof(state))) != NULL);
		td->nt = 4 ;
		assert((td->t = malloc(td->nt * sizeof(transition))) != NULL);

		sprintf(td->s[0].label, "0");
		sprintf(td->s[1].label, "1");
	
		td->s[0].nf = 2;
		td->s[0].nb = 2;
		td->s[1].nf = 2;
		td->s[1].nb = 2;

		td->t[0].i[0] = '0';
		td->t[0].o[0] = '0';
		td->t[0].p = log(gamma);
		td->t[0].os = 0UL;
		td->t[0].ds = 0UL;
		td->s[0].f[0] = 0;
		td->s[0].b[0] = 0;

		td->t[1].i[0] = '1';
		td->t[1].o[0] = '1';
		td->t[1].p = log(1-gamma);
		td->t[1].os = 0UL;
		td->t[1].ds = 1UL;
		td->s[0].f[1] = 1;
		td->s[1].b[0] = 1;

		td->t[2].i[0] = '1';
		td->t[2].o[0] = '1';
		td->t[2].p = log(gamma);
		td->t[2].os = 1UL;
		td->t[2].ds = 1UL;
		td->s[1].f[0] = 2;
		td->s[1].b[1] = 2;

		td->t[3].i[0] = '0';
		td->t[3].o[0] = '0';
		td->t[3].p = log(1-gamma);
		td->t[3].os = 1UL;
		td->t[3].ds = 0UL;
		td->s[1].f[1] = 3;
		td->s[0].b[1] = 3;

		for (j = 0 ; j < 4 ; j++) {
			td->t[j].ni = 1; 
			td->t[j].no = 1; 
			td->t[j].i[1] = '\0';
			td->t[j].o[1] = '\0';
		}
	}
	return (td);
}


/* *****************************************************************************************
   TRELLIS FUNCTIONS
   ***************************************************************************************** */

// three dimensional transducer trellis where the dimensions are
// (ni, no, state), i.e., number of input symbols, number of output
// symbols, and state index of the underlying transducer

int trelcheck(trellis *trel, int ni, int no, unsigned long state)
{
	int md, av;

	if (state > trel->nsund) {
		return (-1);
	}
	if (ni < 0 || ni > trel->nni || no < 0) {
		return (-1);
	}
	md = trel->maxdelta;
	av = trel->av_no[ni];
	if (no < av - md || no > av + md) {
		return (-1);
	}
	return (1);
}

// converts from (input, output, state) to trellis state index
unsigned long trelsind(trellis *trel, int ni, int no, unsigned long state)
{
	int md, av, nni;

	if (trelcheck(trel, ni, no, state) != 1) {
		fprintf(stderr, "Called trelsind with illegal dimensions\n");
		exit(-1);
	}

	md = trel->maxdelta;
	av = trel->av_no[ni];
	return ((ni * (2 * md + 1 + 1) + no - av + md) * trel->nsund + state);

	// note that this "wastes" indices for which av_no[ni]-maxdelta < 0
	// (for small ni) but circumventing this would make addressing
	// unnecessarily complex and probably not worth the memory savings
}

// converts from (input, output, tind) to trellis transition index
// (where tind is the transition index of the underlying transducer)
// (this is not as useful as the other conversion functions because,
// while the states retain their meaning in a trellis, the transitions
// only retain their origin but will likely have new destination states
// in the trellis at other (ni,no) coordinates)
unsigned long treltind(trellis *trel, int ni, int no, unsigned long tind)
{
	int md, av, nni;

	md = trel->maxdelta;
	av = trel->av_no[ni];
	return ((ni * (2 * md + 1 + 1) + no - av + md) * trel->ntund + tind);
}

// extract underlying state index from trellis state index
unsigned long trelstate(trellis *trel, unsigned long nind)
{
	return (nind % trel->nsund);
}

// extract input coordinate from trellis state index
int trelni(trellis *trel, unsigned long nind)
{
	int state, md;
	state = trelstate(trel, nind);
	md = trel->maxdelta;
	return ((((nind - state) / trel->nsund) - (((nind - state) / trel->nsund) % (2 * md + 1 + 1))) / (2 * md + 1 + 1));
}

// extract output coordinate from trellis state index
int trelno(trellis *trel, unsigned long nind)
{
	int av, md, state, ni;
	state = trelstate(trel, nind);
	md = trel->maxdelta;
	ni = trelni(trel, nind);
	av = trel->av_no[ni];
	return ((((nind - state) / trel->nsund) % (2 * md + 1 + 1)) + av - md);
}

trellis *alloc_trellis(unsigned long nsund, unsigned long ntund, int nni, int md, double irate)
{
	trellis *trel;
	transducer *tr;
	int j;

	// allocate memory for the trellis
	assert((trel = malloc(sizeof(trellis))) != NULL);
	trel->tm = NULL;
	trel->fm = NULL;
	trel->bm = NULL;
	trel->bt = NULL;
	assert((trel->td = malloc(sizeof(transducer))) != NULL);
	trel->nsund = nsund;
	trel->ntund = ntund;
	trel->nni = nni;
	trel->maxdelta = md;
	assert((trel->av_no = malloc(nni * sizeof(int))) != NULL);
	for (j = 0 ; j <= nni ; j++)
		trel->av_no[j] = (int)round(j * irate);

	tr = trel->td;
	tr->ns = nsund * (nni + 1) * (2 * md + 2);
	assert((tr->s = malloc(tr->ns * sizeof(state))) != NULL);
	tr->nt = ntund * (nni + 1) * (2 * md + 2);
	assert((tr->t = malloc(tr->nt * sizeof(transition))) != NULL);

	// initialise all state transition counts to 0
	// (they will be set dynamically)
	for (j = 0 ; j < tr->ns ; j++) {
		tr->s[j].label[0] = '\0';
		tr->s[j].nf = 0;
		tr->s[j].nb = 0;
	}

	for (j = 0 ; j < tr->nt ; j++) {
		tr->t[j].os = tr->ns;
		tr->t[j].ds = tr->ns;
		tr->t[j].ni = 0;
		tr->t[j].no = 0;
		tr->t[j].i[0] = '\0';
		tr->t[j].o[0] = '\0';
	}

	return (trel);
}

// build a banded trellis from the transducer provided

trellis * band_trellis(transducer *td, int nni, int md, double irate)
{
	trellis *trel; // trellis
	transducer *tr; // transducer in trellis
	unsigned long j, sind, trtind, tdtind, dsind;
	int k, kk, ni, no, dni, dno, av;

	// allocate memory for the trellis
	trel = alloc_trellis(td->ns, td->nt, nni, md + 1, irate);
	tr = trel->td;

	// compute all the states in the trellis, by copying the
	// underlying transducer for every value of ni and ,
	// then re-connecting the forward transitionsno
	for (ni = 0 ; ni <= nni ; ni++)  { // for all input lengths
		av = trel->av_no[ni];
		for (no = av - md - 1 > 0 ? av - md - 1 : 0 ; no <= av + md + 1 ; no++) // for all output lengths
			for (j = 0 ; j < td->ns ; j++) { // for all
				sind = trelsind(trel, ni, no, j); // index of trellis state
				sprintf(tr->s[sind].label, "%s_%d/%d", td->s[j].label, ni, no);
				tr->s[sind].nf = td->s[j].nf; // copy number of forward links from transducer
				for (k = 0 , kk = 0 ; k < td->s[j].nf ; k++, kk++) { // for all forward transitions
					tdtind = td->s[j].f[k]; // index of underlying transducer forward transition
					dni = ni + td->t[tdtind].ni; // compute destination state tape coordinates
					dno = no + td->t[tdtind].no;
					if (trelcheck(trel, dni, dno, 0) != 1) { // just check legality of dni/dno (state irrelevant)
						//	  if (dni < 0 || dni >= nni || dno < 0 ||
						//	      dno <= trel->av_no[dni]-md || dno >= trel->av_no[dni]+md) {
						// delete this forward transition (it points out of the time window of interest)
						tr->s[sind].nf--; // decrease number of transitions
						kk--; // rewind trellis transition counter by one
						continue; // skip transition connecting steps
					}
					// compute destination state and connect transition
					trtind = treltind(trel, ni, no, tdtind); // index of trellis forward transition
					dsind = trelsind(trel, dni, dno, td->t[tdtind].ds); // destination state index
					memcpy(&(tr->t[trtind]), &(td->t[tdtind]), sizeof(transition)); // copy transition data from transducer
					tr->s[sind].f[kk] = trtind; // set state forward link to transition index
					tr->t[trtind].os = sind; // set transition origin to trellis state address
					tr->t[trtind].ds = dsind;
					assert(tr->s[dsind].nb < MAXT);
					tr->s[dsind].b[tr->s[dsind].nb++] = trtind;
				}
			}
	}
	return (trel);
}

void free_trellis(trellis *trel)
{
	free_transducer(trel->td);
	free(trel->av_no);
	free(trel->tm);
	free(trel->fm);
	free(trel->bm);
	free(trel->bt);
	free(trel);
}

/* *****************************************************************************************
   TRANSDUCER BASED DECODERS
   ***************************************************************************************** */


void viterbi_prep(trellis *trel)
{
	transducer *td;
	td = trel->td;
	assert(trel->tm == NULL);
	assert(trel->fm == NULL);
	assert(trel->bt == NULL);
	assert((trel->tm = malloc(td->nt * sizeof(double))) != NULL);
	assert((trel->fm = malloc(td->ns * sizeof(double))) != NULL);
	assert((trel->bt = malloc(td->ns * sizeof(unsigned long))) != NULL);
}

// char* viterbi(trellis *trel, char *y, unsigned long initstate, unsigned long endstate)
int viterbi(trellis *trel, char *y, unsigned long istate, unsigned long estate, int nni, char *decoded, unsigned long *passes)
{
	transducer *tr;
	int ni, no, no_ds, no_os, nno, passflag;
	unsigned long j, initstate, endstate;

	decoded[0] = '\0';

	nno = strlen(y);
	if (trelcheck(trel, nni, nno, estate) != 1) {
		//    fprintf(stderr, "End state out of banded trellis range, trace=%s\n", trace);
		return (-1); // received length incompatible with banded trellis, decoding impossible
	}

	endstate = trelsind(trel, nni, nno, estate);
	initstate = trelsind(trel, 0, 0, istate);

	tr = trel->td;

	for (j = 0 ; j < tr->ns ; j++) {
		trel->fm[j] = INFINITY;
		trel->bt[j] = tr->nt;
	}
	trel->fm[initstate] = 0.0;


	for (j = 0 ; j < tr->nt ; j++) {
		no = trelno(trel, tr->t[j].os);
		ni = trelni(trel, tr->t[j].os);
		if ((no + tr->t[j].no <= nno) &&
		        (memcmp(y + no, tr->t[j].o, tr->t[j].no * sizeof(char)) == 0)) {
			trel->tm[j] = -tr->t[j].p; // minus as because it is seen as a weight that we want to minimize
		}
		else
			trel->tm[j] = INFINITY;
	}

	*passes = 0UL;
	do {
		passflag = 0;
		(*passes)++;
		for (j = 0 ; j < tr->nt ; j++)
			if (trel->fm[tr->t[j].os] + trel->tm[j] < trel->fm[tr->t[j].ds]) {
				trel->fm[tr->t[j].ds] = trel->fm[tr->t[j].os] + trel->tm[j];
				trel->bt[tr->t[j].ds] = j;
				passflag = 1;
			}
	} while (passflag == 1);
	(*passes)--;

	if (trel->bt[endstate] == tr->nt) { // end state not reached
		//    fprintf(stderr, "End state not reached, trace=%s\n", trace);
		return (-2);
	}

	for (j = endstate ; j != initstate ; j = tr->t[trel->bt[j]].os) {
		ni = trelni(trel, tr->t[trel->bt[j]].os);
		memcpy(decoded + ni, tr->t[trel->bt[j]].i, tr->t[trel->bt[j]].ni * sizeof(char));
	}

	decoded[nni] = '\0';
	return (1);
	//  return(decoded);
}

void free_viterbi(trellis *trel)
{
	free(trel->tm);
	free(trel->fm);
	free(trel->bt);
	trel->tm = NULL;
	trel->fm = NULL;
	trel->bt = NULL;
}

void bcjr_prep(trellis *trel)
{
	int k, nni;
	transducer *td;
	td = trel->td;
	assert(trel->tm == NULL);
	assert(trel->fm == NULL);
	assert(trel->bm == NULL);
	assert((trel->tm = malloc(td->nt * sizeof(double))) != NULL);
	assert((trel->fm = malloc(td->ns * sizeof(double))) != NULL);
	assert((trel->bm = malloc(td->ns * sizeof(double))) != NULL);
	//nni = trelni(trel, (td->ns)-1);
}

double max_star(double a, double b) {
	double c;
	if (a > b) c = a + log(1 + exp(-fabs(a - b)));
	else c = b + log(1 + exp(-fabs(a - b)));
	if (a == -INFINITY) c = b;
	else if (b == -INFINITY) c = a;
	return (c);
}

int bcjr(trellis *trel, char *y, char *x, unsigned long istate, unsigned long estate, int nni, double *Lin, unsigned long *passes, int x_not_given, double *log_py)
{
	transducer *tr;
	int ni, no, nno, passflag;
	unsigned long j, jj, k, initstate, endstate;
	double previous_metric, p_1_input, sum_1, sum_2;

	nno = strlen(y);
	if (trelcheck(trel, nni, nno, estate) != 1)
		return (-1); // received length incompatible with banded trellis, decoding impossible
	tr = trel->td;
	for (j = 0 ; j < tr->ns ; j++) {
		trel->fm[j] = -INFINITY;
		trel->bm[j] = -INFINITY;
	}

	initstate = trelsind(trel, 0, 0, istate);
	trel->fm[initstate] = 0.0;
	//if (estate == trel->nsund) { // no termination
	//	for (j = 0 ; j < trel->nsund ; j++) {
	//		endstate = trelsind(trel, nni, nno, j);
	//		trel->bm[endstate] = 1.0 / trel->nsund;
	//	}
	//}
	endstate = trelsind(trel, nni, nno, estate);
	trel->bm[endstate] = 0.0;

	for (j = 0 ; j < tr->nt ; j++) {
		no = trelno(trel, tr->t[j].os);
		ni = trelni(trel, tr->t[j].os);
		if ((no + tr->t[j].no <= nno) &&
		        ((memcmp(y + no, tr->t[j].o, tr->t[j].no * sizeof(char)) == 0))&&
		        ((memcmp(x + ni, tr->t[j].i, tr->t[j].ni * sizeof(char)) == 0) || x_not_given))
			trel->tm[j] = tr->t[j].p;
		else
			trel->tm[j] = -INFINITY;

		if (tr->t[j].ni == 1) {
			if (Lin != NULL) {
				ni = trelni(trel, tr->t[j].os);
				if (tr->t[j].i[0] == '0')
					trel->tm[j] -= log((1 + exp(-Lin[ni])));
				else
					trel->tm[j] -= log(1 + exp(Lin[ni]));
			}
		}
	}
	/*
	//Normalisation
	for (j = 0 ; j < tr->ns ; j++) {
		//Find the probability of a transition with no input + sum of the probabilities 
		//(unnormalised) of teh transitions with 1 input
		sum_1 = -INFINITY;
		sum_2 = -INFINITY;
		for (k = 0 ; k < tr->s[j].nf ; k++) {
			if (tr->t[tr->s[j].f[k]].ni == 1 )
			{
				sum_1 =  max_star(sum_1, trel->tm[tr->s[j].f[k]]);
			}
			else {
				sum_2 = max_star(sum_2, trel->tm[tr->s[j].f[k]]);
			}
		}

		//probability of a transition with 1 input
		p_1_input = log(1-exp(sum_2));
		if (sum_1 != -INFINITY) {
			for (k = 0 ; k < tr->s[j].nf ; k++) {
				if (tr->t[tr->s[j].f[k]].ni == 1 )
				{
					trel->tm[tr->s[j].f[k]] += p_1_input - sum_1;
				}
			}
		}
	} 
	*/
	*passes = 0UL;
	do {
		passflag = 0;
		(*passes)++;
		for (j = 0 ; j < tr->ns ; j++) {
			jj = tr->ns - j - 1; // reverse count just for speed otherwise will result in too many passes
			previous_metric = trel->fm[j];
			trel->fm[j] = -INFINITY;
			for (k = 0 ; k < tr->s[j].nb ; k++) {
				trel->fm[j] = max_star(trel->tm[tr->s[j].b[k]] + trel->fm[tr->t[tr->s[j].b[k]].os], trel->fm[j]);
			}
			if (trel->fm[j] > previous_metric) {
				passflag = 1;
			}
			else
				trel->fm[j] = previous_metric;
			previous_metric = trel->bm[jj];
			trel->bm[jj] = -INFINITY;
			for (k = 0 ; k < tr->s[jj].nf ; k++) {
				trel->bm[jj] = max_star(trel->tm[tr->s[jj].f[k]] + trel->bm[tr->t[tr->s[jj].f[k]].ds], trel->bm[jj]);
			}
			if (trel->bm[jj] > previous_metric) {
				passflag = 1;
			}
			else
				trel->bm[jj] = previous_metric;
		}
	} while (passflag == 1);
	(*passes)--;
	if (log_py != NULL)
		*log_py = trel->fm[endstate];
	return (1);
}

void bcjr_inputprobs(trellis *trel, double *L)
{
	double log_p0[MAX_TRANSMIT_LENGTH], log_p1[MAX_TRANSMIT_LENGTH];
	unsigned long j;
	int ni;
	transducer *tr;

	tr = trel->td;

	for (j = 0 ; j < trel->nni ; j++) {
		log_p0[j] = -INFINITY;
		log_p1[j] = -INFINITY;
	}

	for (j = 0 ; j < tr->nt ; j++) {
		if (tr->t[j].ni != 1)
			continue;
		ni = trelni(trel, tr->t[j].os);
		if (ni >= trel->nni)
			continue;
		if (tr->t[j].i[0] == '0') {
			log_p0[ni] = max_star(trel->fm[tr->t[j].os] + trel->tm[j] + trel->bm[tr->t[j].ds], log_p0[ni]);
		}
		else {
			log_p1[ni] = max_star(trel->fm[tr->t[j].os] + trel->tm[j] + trel->bm[tr->t[j].ds], log_p1[ni]);
		}
	}
	for (j = 0 ; j < trel->nni ; j++) {
		L[j] = log_p0[j] - log_p1[j];
	}
}

void free_bcjr(trellis *trel)
{
	free(trel->tm);
	free(trel->fm);
	free(trel->bm);
	trel->tm = NULL;
	trel->fm = NULL;
	trel->bm = NULL;
}

/* *****************************************************************************************
   SIMPLE RATE 1/2 OR 1 CONVOLUTIONAL CODE IMPLEMENTATION FOR TESTING PURPOSES

   NOTE: internally, these functions use an LSB notation to describe the convolutional
   code polynomials, e.g., polynomial 015, binary 1101 corresponds to 1+D+D^3, so that
   the connections are to u_k, u_{k-1} and u_{k-3}. MATLAB and other sources use a MSB
   notation, where 1101 is interpreted as D^3+D^2+1, i.e.,, the connections are to u_k,
   u_{k-2} and u_{k-3} for polyonmial 015. Use the conv_convert function to convert from
   MATLAB notation to our notation (or vice versa).
   ***************************************************************************************** */

// convert from p1i,p2i to p1 and p2

void conv_convert(unsigned long p1i, unsigned long p2i, unsigned long *p1, unsigned long *p2)
{
	unsigned long p1o = 0UL, p2o = 0UL;
	int mem, k;

	for (mem = 0 ; !((p1i >> mem == 0) && (p2i >> mem == 0)) ; mem++);

	for (k = 0 ; k < mem ; k++) {
		p1o |= ((p1i >> k) & 1UL);
		p2o |= ((p2i >> k) & 1UL);
		p1o <<= 1;
		p2o <<= 1;
	}
	*p1 = p1o >> 1;
	*p2 = p2o >> 1;
}

int hamming_weight(unsigned long x)
{
	int k;
	for (k = 0 ; x != 0 ; k += (x & 1UL) , x >>= 1) ;
	return (k);
}

// rate 1/2 or 1 convolutional encoder (if p2 =0, rate 1, otherwise rate 1/2)
long unsigned cc(char *x, unsigned long p1, unsigned long p2, char *y, int rec)
{
	int N, j, mem;
	unsigned long state, out_bit, k, last_state_message_bit;

	N = strlen(x);
	mem = log2int(p1) > log2int(p2) ? log2int(p1) : log2int(p2);
	mem--;
	for (j = 0 , state = 0UL; j < N ; j++) {
		if (rec == 0) {
			state = (state << 1) | (x[j] - '0');
			if (p2 != 0) {
				y[2 * j] = '0' + (hamming_weight(state & p1) & 1UL);
				y[2 * j + 1] = '0' + (hamming_weight(state & p2) & 1UL);
			}
			else y[j] = '0' + (hamming_weight(state & p1) & 1UL);
		}
		else {
			out_bit = (hamming_weight((state << 1)&p1) & 1UL);
			state = (state << 1) | (out_bit ^ (x[j] - '0'));
			if (p2 != 0) {
				y[2 * j] = '0' + (hamming_weight(state & (p2)) & 1UL);
				y[2 * j + 1] = x[j];
			}
			else y[j] = '0' + (out_bit ^ (x[j] - '0'));
		}
		if (j == N - mem - 1) last_state_message_bit = state;
	}
	if (p2 != 0) y[2 * N] = '\0';
	else y[N] = '\0';

	// it returns the state after sending the last bit of the message
	// the later bits are just sent to ensure that the final state is all zeros
	return (last_state_message_bit);
}


transducer *conv2transducer(unsigned long p1, unsigned long p2, int rec)
{
	transducer *td;
	int mem;
	unsigned long j, k, n, dest_0, dest_1, out_bit;
	// compute memory of conv code according to connection polynomials
	for (mem = 0 ; !((p1 >> mem == 0) && (p2 >> mem == 0)) ; mem++);
	mem--;
	// allocate space and dimensions for transducer
	assert((td = malloc(sizeof(transducer))) != NULL);
	td->ns = 1 << mem;
	assert((td->s = malloc(td->ns * sizeof(state))) != NULL);
	td->nt = td->ns << 1;
	assert((td->t = malloc(td->nt * sizeof(transition))) != NULL);

	// initialise all backward transition counts to zero (so they
	// can be set dynamically below)
	for (j = 0UL ; j < td->ns ; j++)
		td->s[j].nb = 0;
	// now compute all forward transitions from every state and
	// connect as appropriate
	for (j = 0UL ; j < td->ns ; j++) {
		// label the state with the binary content of shift reg
		// MSB to LSB left to right!
		for (k = 0 ; k < mem ; k++)
			if (((j >> k) & 1UL) == 1UL)
				td->s[j].label[mem - k - 1] = '1';
			else
				td->s[j].label[mem - k - 1] = '0';
		td->s[j].label[mem] = '\0'; // terminate string
		if (mem == 0) {
			td->s[j].label[0] = '0';
			td->s[j].label[1] = '\0';
		}
		// connect two forward transitions for 0 and 1 input
		td->s[j].nf = 2;
		td->s[j].f[0] = j << 1; // 0 input transition index
		td->s[j].f[1] = (j << 1) | 1UL; // 1 input transition index
		// connect transition origin states to current state
		td->t[j << 1].os = j;
		td->t[(j << 1) | 1UL].os = j;
		// compute destination states (shift 0 and 1 into register)
		// (we shift 0 and the inde of the 1 shift is ind0+1)
		if (mem == 0) {
			td->t[0].ds = 0;
			td->t[1].ds = 0;
			td->s[0].b[0] = 0;
			td->s[0].b[1] = 1;
		}
		else {
			k = (j << 1) & ((1UL << mem) - 1);
			if (rec == 0) {
				// connect transition destination states to 0 and 1 shifted states
				td->t[j << 1].ds = k;
				td->t[(j << 1) | 1UL].ds = k | 1UL;
				// connect backward transitions of both destination states
				td->s[k].b[td->s[k].nb++] = j << 1;
				td->s[k | 1UL].b[td->s[k | 1UL].nb++] = (j << 1) | 1UL;
			}
			else {
				out_bit = (hamming_weight((j << 1)&p1) & 1UL);
				// connect transition destination states to 0 and 1 shifted states
				td->t[j << 1].ds = ((j << 1) | out_bit) % (1UL << mem);
				td->t[(j << 1) | 1UL].ds = ((j << 1) | (out_bit ^ 1UL)) % (1UL << mem);
				// connect backward transitions of both destination states
				td->s[td->t[j << 1].ds].b[td->s[td->t[j << 1].ds].nb++] = j << 1;
				td->s[td->t[(j << 1) | 1UL].ds].b[td->s[td->t[(j << 1) | 1UL].ds].nb++] = (j << 1) | 1UL;
			}
		}

		td->t[j << 1].p = 0; // given the input, the transition is deterministic (p = 1)
		td->t[(j << 1) | 1UL].p = 0; // given the input, the transition is deterministic (p = 1)
		// set input alphabet length of transitions to 1
		td->t[j << 1].ni = 1;
		td->t[(j << 1) | 1UL].ni = 1;
		// set output alphabet length of transitions to 1 or 2 depending on p2
		if (p2 != 0) {
			td->t[j << 1].no = 2;
			td->t[(j << 1) | 1UL].no = 2;
		}
		else {
			td->t[j << 1].no = 1;
			td->t[(j << 1) | 1UL].no = 1;
		}

		// set input symbol for transitions to 0 and 1 respectively
		strcpy(td->t[j << 1].i, "0");
		strcpy(td->t[(j << 1) | 1UL].i, "1");
		// compute output symbols for each transition
		if (rec == 0) {
			td->t[j << 1].o[0] = '0' + (hamming_weight((j << 1)&p1) & 1UL);
			if (p2 != 0) {
				td->t[j << 1].o[1] = '0' + (hamming_weight((j << 1)&p2) & 1UL);
				td->t[j << 1].o[2] = '\0';
			}
			else td->t[j << 1].o[1] = '\0';
			td->t[(j << 1) | 1UL].o[0] = '0' + (hamming_weight(((j << 1) | 1UL)&p1) & 1UL);
			if (p2 != 0) {
				td->t[(j << 1) | 1UL].o[1] = '0' + (hamming_weight(((j << 1) | 1UL)&p2) & 1UL);
				td->t[(j << 1) | 1UL].o[2] = '\0';
			}
			else td->t[(j << 1) | 1UL].o[1] = '\0';
		}
		else {
			if (p2 != 0) {
				td->t[j << 1].o[0] = '0' + (hamming_weight(((j << 1) | out_bit)&p2) & 1UL);
				td->t[j << 1].o[1] = '0';
				td->t[j << 1].o[2] = '\0';
			}
			else {
				td->t[j << 1].o[0] = '0' + out_bit;
				td->t[j << 1].o[1] = '\0';
			}

			if (p2 != 0) {
				td->t[(j << 1) | 1UL].o[0] = '0' + (hamming_weight(((j << 1) | (out_bit ^ 1UL))&p2) & 1UL);
				td->t[(j << 1) | 1UL].o[1] = '1';
				td->t[(j << 1) | 1UL].o[2] = '\0';
			}
			else {
				td->t[(j << 1) | 1UL].o[0] = '0' + (out_bit ^ 1UL);
				td->t[(j << 1) | 1UL].o[1] = '\0';
			}
		}
	}
	return (td);
}

/* *****************************************************************************************
   INSERTION DELETION CHANNEL
   ***************************************************************************************** */

transducer *insdel2transducer(double pi, double pd, double ps)
{
	transducer *td;
	unsigned long j;

	assert((td = malloc(sizeof(transducer))) != NULL);
	td->ns = 1;
	assert((td->s = malloc(sizeof(state))) != NULL);
	td->nt = 8 ;
	assert((td->t = malloc(td->nt * sizeof(transition))) != NULL);

	sprintf(td->s[0].label, "0");
	td->s[0].nf = td->nt;
	td->s[0].nb = td->nt;
	for (j = 0 ; j < td->nt ; j++) {
		td->s[0].f[j] = j;
		td->s[0].b[j] = j;
		td->t[j].os = 0UL;
		td->t[j].ds = 0UL;
		td->t[j].ni = 1; // default (will be overridden for insertions)
		td->t[j].no = 1; // default (will be overridden for deletions)
		td->t[j].i[1] = '\0';
		td->t[j].o[1] = '\0';
	}
	td->t[0].i[0] = '0';
	td->t[0].o[0] = '0';
	td->t[0].p = log(1-ps-pi-pd);
	//td->t[0].p = 0;
	td->t[1].i[0] = '1';
	td->t[1].o[0] = '1';
	td->t[1].p = log(1-ps-pi-pd);
	//td->t[1].p = 0;
	td->t[2].i[0] = '0';
	td->t[2].o[0] = '1';
	td->t[2].p = log(ps);
	//td->t[2].p = 3;
	td->t[3].i[0] = '1';
	td->t[3].o[0] = '0';
	td->t[3].p = log(ps);
	//td->t[3].p = 3;
	td->t[4].i[0] = '0';
	td->t[4].o[0] = '\0';
	td->t[4].no = 0;
	td->t[4].p = log(pd);
	//td->t[4].p = 2;
	td->t[5].i[0] = '1';
	td->t[5].o[0] = '\0';
	td->t[5].no = 0;
	td->t[5].p = log(pd);
	//td->t[5].p = 2;
	td->t[6].i[0] = '\0';
	td->t[6].o[0] = '0';
	td->t[6].ni = 0;
	td->t[6].p = log(pi/2);
	//td->t[6].p = 1;
	td->t[7].i[0] = '\0';
	td->t[7].o[0] = '1';
	td->t[7].ni = 0;
	td->t[7].p = log(pi/2);
	//td->t[7].p = 1;

	return (td);
}


// insertion deletion channel
// char *insdelch(char *x, double pi, double pd, double ps)
void insdelch(char *x, double pi, double pd, double ps, char *y, int maxNy, char *trace)
{
	//  char *y;
	int N, Ny = 0, kx = 0, ky = 0;
	double rand;
	int trk = 0;

	N = strlen(x);
	while (1) {
		rand = genrand_real1();
		if (kx == N && rand >= pi) // if input vector used up and no more insertions expected
			break;

		switch ((rand >= pi) + (rand >= pi + pd) + (rand >= pi + pd + ps)) {
		case 0: // insertion
			if (genrand_real1() < .5)
				y[ky++] = '0';
			else
				y[ky++] = '1';
			trace[trk++] = 'i';
			break;
		case 1: // deletion
			kx++;
			trace[trk++] = 'd';
			break;
		case 2: // substitution
			if (x[kx++] == '0')
				y[ky++] = '1';
			else
				y[ky++] = '0';
			trace[trk++] = 's';
			break;
		case 3:
		default: // transmission
			y[ky++] = x[kx++];
			trace[trk++] = 't';
		}
		if (ky >= maxNy) {
			printf("insdelch exceeded output length, increase maxNy!\n");
			exit(-1);
		}
	}
	y[ky] = '\0';
	trace[trk] = '\0';
}

/* *****************************************************************************************
   LDPC CODES
   ***************************************************************************************** */

void print_matrix(sparse_matrix *matrix, int mode) {
	//mode 0 is normal mode e.g. for the parity check matrix
	//mode 1 is when the matrix was transposed e.g. for the generator matrix
	int k, l, flag, m = 0;
	if (mode == 0) {
		printf("\n");
		for (k = 0; k < matrix->M; k++) {
			for (l = 0; l < matrix->N; l++) {
				if (matrix->rows[m] == l + 1) {
					printf("%d ", 1);
					m++;
				}
				else printf("%d ", 0);
			}
			printf("\n");
		}
	}
	else {
		printf("\n");
		for (k = 0; k < matrix->M; k++) {
			for (l = 0; l < matrix->N; l++) {
				flag = 0;
				if (l == 0) {
					for (m = 0; m < matrix->ones_per_row[0]; m++) {
						if (matrix->rows[m] == k + 1) {
							flag = 1;
							break;
						}
					}
				}
				else {
					for (m = matrix->ones_per_row[l - 1]; m < matrix->ones_per_row[l]; m++) {
						if (matrix->rows[m] == k + 1) {
							flag = 1;
							break;
						}
					}
				}
				if (flag == 1) printf("%d ", 1);
				else printf("%d ", 0);
			}
			printf("\n");
		}
	}
}

int count_number_bit_errors (char *u, char * dec, int length) {
	int k, count = 0;
	for (k = 0; k < length; k++) {
		if (u[k] != dec[k])count++;
	}
	return (count);
}


sparse_matrix * load_parity_matrix2(char * file_name)
{
	int k, l, m, value, max_ones_per_row;
	sparse_matrix *data;
	assert((data = malloc(sizeof(sparse_matrix))) != NULL);
	FILE *f;
	f = fopen(file_name , "r");
	assert(fscanf(f, "%d", &data->N) == 1);
	assert(fscanf(f, "%d", &data->M) == 1);
	if (data->M > MAX_ROWS) {
		fprintf(stderr, "Too many rows\n");
		exit(-1);
	}
	assert(fscanf(f, "%d", &max_ones_per_row) == 1);
	assert((data-> ones_per_row = malloc(data->M * sizeof(int))) != NULL);
	for (k = 0 ; k < data->M ; k++) {
		m = 0;
		for (l = 0 ; l < max_ones_per_row ; l++) {
			assert(fscanf(f, "%d", &value) == 1);
			if (value != 0) m++;
		}

		if (k == 0) data->ones_per_row[k] = m;
		else data->ones_per_row[k] = data->ones_per_row[k - 1] + m;
	}
	assert((data->rows = malloc(data->ones_per_row[data->M - 1] * sizeof(int))) != NULL);
	fclose(f);
	f = fopen(file_name , "r");
	assert(fscanf(f, "%d", &data->N) == 1);
	assert(fscanf(f, "%d", &data->M) == 1);
	assert(fscanf(f, "%d", &max_ones_per_row) == 1);
	m = 0;
	for (k = 0 ; k < data->M ; k++) {
		for (l = 0 ; l < max_ones_per_row ; l++) {
			assert(fscanf(f, "%d", &value) == 1);
			if (value != 0) {
				data->rows[m] = value;
				m++;
			}
		}
	}
	fclose(f);
	return (data);
}

sparse_matrix ** load_2_matrixes(char * file_name)
{
	int k, l, m, value, max_ones_per_row, max_ones_per_col, rows_par;
	sparse_matrix **matrixes;
	sparse_matrix *gen; // for the generator matrix, columns and rows are swaped (see definition of the sparse_matrix struct)
	sparse_matrix *par;

	assert((matrixes = malloc(2 * sizeof(sparse_matrix*))) != NULL);

	assert((gen = malloc(sizeof(sparse_matrix))) != NULL);
	assert((par = malloc(sizeof(sparse_matrix))) != NULL);

	FILE *f;
	f = fopen(file_name , "r");
	assert(f);
	assert(fscanf(f, "%d", &par->N) == 1);
	gen->N = par->N;
	assert(fscanf(f, "%d", &gen->M) == 1);
	assert(fscanf(f, "%d", &par->M) == 1);
	assert(fscanf(f, "%d", &max_ones_per_col) == 1);
	assert(fscanf(f, "%d", &max_ones_per_row) == 1);
	assert((gen-> ones_per_row = malloc(gen->N * sizeof(int))) != NULL);
	for (l = 0 ; l < gen->N ; l++) {
		gen-> ones_per_row[l] = max_ones_per_col;
	}
	for (k = 0 ; k <  max_ones_per_col; k++) {
		for (l = 0 ; l < gen->N ; l++) {
			assert(fscanf(f, "%d", &value) == 1);
			if (value == 0 && gen-> ones_per_row[l] == max_ones_per_col) gen->ones_per_row[l] = k;
		}
	}
	for (l = 0 ; l < gen->N ; l++) {
		if (l > 0) gen->ones_per_row[l] = gen->ones_per_row[l] + gen->ones_per_row[l - 1];
	}
	assert((gen->rows = malloc(gen->ones_per_row[gen->N - 1] * sizeof(int))) != NULL);
	assert((par-> ones_per_row = malloc(par->M * sizeof(int))) != NULL);
	for (k = 0 ; k < par->M ; k++) {
		m = 0;
		for (l = 0 ; l < max_ones_per_row ; l++) {
			assert(fscanf(f, "%d", &value) == 1);
			if (value != 0) m++;
		}
		if (k == 0) par->ones_per_row[k] = m;
		else par->ones_per_row[k] = par->ones_per_row[k - 1] + m;
	}
	assert((par->rows = malloc(par->ones_per_row[par->M - 1] * sizeof(int))) != NULL);
	fclose(f);

	f = fopen(file_name , "r");
	assert(fscanf(f, "%d", &par->N) == 1);
	gen->N = par->N;
	assert(fscanf(f, "%d", &gen->M) == 1);
	assert(fscanf(f, "%d", &par->M) == 1);
	assert(fscanf(f, "%d", &max_ones_per_col) == 1);
	assert(fscanf(f, "%d", &max_ones_per_row) == 1);

	for (k = 0 ; k <  max_ones_per_col; k++) {
		for (l = 0 ; l < gen->N ; l++) {
			assert(fscanf(f, "%d", &value) == 1);
			if (value != 0) {
				if (l == 0) gen->rows[k] = value;
				else gen->rows[gen->ones_per_row[l - 1] + k] = value;
			}
		}
	}
	m = 0;
	for (k = 0 ; k < par->M ; k++) {
		for (l = 0 ; l < max_ones_per_row ; l++) {
			assert(fscanf(f, "%d", &value) == 1);
			if (value != 0) {
				par->rows[m] = value;
				m++;
			}
		}
	}

	fclose(f);
	matrixes[0] = gen;
	matrixes[1] = par;
	return (matrixes);
}

void free_sparse_matrix(sparse_matrix * data) {
	free(data->ones_per_row);
	free(data->rows);
	free(data);
}

Tanner_graph *Generate_Tanner_graph (sparse_matrix * parity_check_matrix) {
	int k, l, m, n;
	Tanner_graph *graph;

	assert((graph = malloc(sizeof(Tanner_graph))) != NULL);
	graph->N = parity_check_matrix->N;
	graph->M = parity_check_matrix->M;
	graph->number_of_ones = parity_check_matrix->ones_per_row[parity_check_matrix->M - 1];
	assert((graph-> col_connect = malloc(graph->N * sizeof(int))) != NULL);
	assert((graph-> row_connect = malloc(graph->M * sizeof(int))) != NULL);
	assert((graph-> interleaver = malloc(graph->number_of_ones * sizeof(int))) != NULL);
	memset(graph->interleaver, 0, graph->number_of_ones * sizeof(int));
	memset(graph->col_connect, 0, graph->N * sizeof(int));
	for (k = 0 ; k < graph->M; k++) graph->row_connect[k] = parity_check_matrix->ones_per_row[k];
	for (k = 0 ; k < graph->number_of_ones; k++)
		graph->col_connect[parity_check_matrix->rows[k] - 1]++;
	for (k = 1 ; k < graph->N; k++) {
		graph->col_connect[k] = graph->col_connect[k] + graph->col_connect[k - 1];
	}
	m = 1;
	for (k = 0 ; k < graph->number_of_ones; k++) {
		n = 0;
		while (1) {
			if (parity_check_matrix->rows[k] == 1) {
				if (graph->interleaver[n] == 0) {
					graph->interleaver[n] = m;
					break;
				}
			}
			else {
				if (graph->interleaver[graph->col_connect[parity_check_matrix->rows[k] - 2] + n] == 0) {
					graph->interleaver[graph->col_connect[parity_check_matrix->rows[k] - 2] + n] = m;
					break;
				}
			}
			n++;
		}
		m++;
	}

	for (k = 0 ; k < graph->col_connect[graph->N - 1]; k++) graph->interleaver[k] = graph->interleaver[k] - 1;
	return (graph);
}


void free_Tanner_graph(Tanner_graph * graph) {
	free(graph->col_connect);
	free(graph->row_connect);
	free(graph->interleaver);
	free(graph);
}

void LDPC_encode_from_gen(char *x, sparse_matrix *matrix, char *y)
{
	//In this function rows means columns (see definition struct sparse_matrix)
	int k, j, l;
	for (k = 0 ; k < matrix->N ; k++) {
		j = 0;
		if (k == 0) {
			for (l = 0; l < matrix->ones_per_row[0]; l++) {
				if (x[matrix->rows[l] - 1] == '1') j++;
			}
		}
		else {
			for (l = matrix->ones_per_row[k - 1]; l < matrix->ones_per_row[k]; l++) {
				if (x[matrix->rows[l] - 1] == '1') j++;
			}
		}
		j = j % 2;
		y[k] = '0' + j;
	}
	y[matrix->N] = '\0';
}

void BSC_LLR(char *y, double ps, double * LLR) {
	int k, N;
	N = strlen(y);
	for (k = 0; k < N; k++) {
		if (y[k] == '0') LLR[k] = log((1 - ps) / ps);
		else LLR[k] = -log((1 - ps) / ps);
	}
}
void hard_decision(double * LLR, char * decoded, int init, int end) {
	int k, l = 0;
	for (k = init; k < end ; k++, l++) {
		if (LLR[k] > 0.0) decoded[l] = '0';
		else decoded[l] = '1';
	}
}

int LDPC_codeword_found (sparse_matrix * matrix, double * LLR) {
	int k, l, sum = 0;
	for (k = 0; k < matrix->M; k++) {
		if (k == 0) {
			for (l = 0; l < matrix->ones_per_row[0]; l++) {
				if (LLR[matrix->rows[l] - 1] < 0) sum++;
			}
			sum = sum % 2;
			if (sum == 1) return (0);
		}
		else {
			for (l = matrix->ones_per_row[k - 1]; l < matrix->ones_per_row[k]; l++) {
				if (LLR[matrix->rows[l] - 1] < 0) sum++;
			}
			sum = sum % 2;
			if (sum == 1) return (0);
		}
	}
	return (1);
}

int is_LDPC_stuck (double * previous_LLR, double * LLR, int length) {
	int k, l, sum = 0;
	for (k = 0; k < length; k++) {
		if (fabs(LLR[k] - previous_LLR[k]) > EPSILON) return (0);
	}
	return (1);
}

void LDPC_var(Tanner_graph *graph, double *messages_cv, double *messages_vc,double *LLR_0, double *LLR) {
	int j, k, l;
	for (k = 0; k < graph->N; k++) {
		if (k == 0) {
			for (j = 0; j < graph->col_connect[0]; j++) {
				messages_vc[graph->interleaver[j]] = LLR_0[k];
				for (l = 0; l < graph->col_connect[0]; l++)
					if (j != l) messages_vc[graph->interleaver[j]] =
							messages_vc[graph->interleaver[j]] + messages_cv[graph->interleaver[l]];
			}
		}
		else {
			for (j = graph->col_connect[k - 1]; j < graph->col_connect[k]; j++) {
				messages_vc[graph->interleaver[j]] = LLR_0[k];
				for (l = graph->col_connect[k - 1]; l < graph->col_connect[k]; l++)
					if (j != l) messages_vc[graph->interleaver[j]] =
							messages_vc[graph->interleaver[j]] + messages_cv[graph->interleaver[l]];
			}
		}
		LLR[k] = messages_vc[graph->interleaver[j - 1]] + messages_cv[graph->interleaver[l - 1]];
	}
}

void LDPC_check(Tanner_graph *graph, double *messages_cv, double *messages_vc,double *LLR_0, double *LLR) {
	int j, k, l;
	for (k = 0; k < graph->M; k++) {
		if (k == 0) {
			for (j = 0; j < graph->row_connect[0]; j++) {
				messages_cv[j] = 1.0;
				for ( l = 0; l < graph->row_connect[0]; l++)
					if (l != j) messages_cv[j] = messages_cv[j] * tanh(0.5 * messages_vc[l]);
				messages_cv[j] = 2 * atanh(messages_cv[j]);
			}
		}
		else {
			for (j = graph->row_connect[k - 1]; j < graph->row_connect[k]; j++) {
				messages_cv[j] = 1.0;
				for (l = graph->row_connect[k - 1]; l < graph->row_connect[k]; l++)
					if (l != j) messages_cv[j] = messages_cv[j] * tanh(0.5 * messages_vc[l]);
				messages_cv[j] = 2 * atanh(messages_cv[j]);
			}
		}
	}
}

int LDPC_decoder (sparse_matrix * parity_check_matrix, double * LLR) {
	int i, j, k, l;
	double *LLR_0;
	double *previous_LLR;
	double *messages_vc, *messages_cv;
	Tanner_graph *graph;
	graph = Generate_Tanner_graph (parity_check_matrix);

	assert((previous_LLR = malloc(graph->N * sizeof(double))) != NULL);
	memset(previous_LLR, 0, graph->N * sizeof(double));

	assert((LLR_0 = malloc(graph->N * sizeof(double))) != NULL);
	memset(LLR_0, 0, graph->N * sizeof(double));

	assert((messages_cv = malloc(graph->number_of_ones * sizeof(double))) != NULL);
	memset(messages_cv, 0, graph->number_of_ones * sizeof(double));

	assert((messages_vc = malloc(graph->number_of_ones * sizeof(double))) != NULL);
	memset(messages_vc, 0, graph->number_of_ones * sizeof(double));

	for (i = 0; i < graph->N; i++) LLR_0[i] = LLR[i];

	for (i = 0; i < MAX_LDPC_IT; i++) {
		LDPC_var(graph, messages_cv, messages_vc, LLR_0, LLR);
		//if (i > 0 && (is_LDPC_stuck (previous_LLR, LLR, graph->N)
		//	        || LDPC_codeword_found (parity_check_matrix, LLR)))  break;
		for (k = 0; k < graph->N; k++) previous_LLR[k] = LLR[k];
		LDPC_check(graph, messages_cv, messages_vc, LLR_0, LLR);
	}
	free_Tanner_graph(graph);
	return (i);
}

/* *****************************************************************************************
   MARKER CODES
   ***************************************************************************************** */

// d data bits between markers
// m contains 2 consecutive marker sequences in a string of length 2nm where nm is the length of each marker seq
// K data bits in total per codeword
// Rate is d/(d+nm), block length is NK*(d+nm)/d (possibly + a number between 0 and d-1
// if data bits are added after last marker?)
// markers are seleted pseudo-randomly based on the pre-loaded binary sequence randmarkers

trellis* marker_trellis(int d, int nno, int md, double pi, double pd, double ps, int mode)
{
	//mode 1 for iterative decoding
	//mode 0 otherwise (the transducer of th source is then used)
	transducer *tdinsdel, *t_source, *t;
	trellis *trel;

	t_source = Markov_source_transducer(0.5);
	tdinsdel = insdel2transducer(pi,pd,ps);

	t = compose_transducers(t_source, tdinsdel);
	if (mode == 1) trel = band_trellis(t, nno, md, 1);
	else trel = band_trellis(tdinsdel, nno, md, 1);
	free_transducer(tdinsdel);
	free_transducer(t_source);
	free_transducer(t);
	return (trel);
}

void insert_marker_LLR(double *Lin, double *Lout, int nni, int d)
{
	int ni, no, j, mlength, mindex, Nrand;

	Nrand = strlen(randmarkers);
	mlength = strlen(markers) / 2;

	for (ni = 0 , no = 0 , mindex = 0 ; ni < nni ; ) {
		if (ni > 0 && ni % d == 0) { // marker insert location
			for (j = 0 ; j < mlength ; j++)
				Lout[no++] = markers[j + mlength * (randmarkers[mindex % Nrand] - '0')] == '0' ? INFINITY : -INFINITY;
			mindex++;
		}
		Lout[no++] = Lin[ni++];
	}
}

void remove_marker_LLR(double *Lin, double *Lout, int nni, int d)
{
	int ni, no, mlength;
	mlength = strlen(markers) / 2;

	for (ni = 0 , no = 0 ; no < nni ; ) {
		if (no > 0 && no % d == 0)
			ni += mlength;
		Lout[no++] = Lin[ni++];
	}
}

void marker_encode(char *in, char *out, int d)
{
	int j, ni, no, mlength, mindex, nni, Nrand;

	nni = strlen(in);
	Nrand = strlen(randmarkers);
	mlength = strlen(markers) / 2;

	for (ni = 0, no = 0, mindex = 0 ; ni < nni ; ) {
		if (ni > 0 && ni % d == 0) { // insert marker
			for (j = 0 ; j < mlength ; j++)
				out[no++] = markers[j + mlength * (randmarkers[mindex % Nrand] - '0')];
			mindex++;
		}
		out[no++] = in[ni++];
	}
	if (ni > 0 && ni % d == 0) { // insert marker
			for (j = 0 ; j < mlength ; j++)
				out[no++] = markers[j + mlength * (randmarkers[mindex % Nrand] - '0')];
			mindex++;
		} 
	out[no] = '\0';
}

/* *****************************************************************************************
   UTILITY FUNCTIONS
   ***************************************************************************************** */
int check_if_inf (double *LLR, int length) {
	int k;
	for (k = 0; k < length; k++)
		if (LLR[k] == INFINITY || LLR[k] == -INFINITY)
			return (1);
	return (0);
}

int log2int(unsigned long x)
{
	int k;
	for (k = 0 ; x != 0UL ; k++, x >>= 1);
	return (k);
}

// binary symmetric source
void bss(int N, char *x)
{
	int k;
	for (k = 0 ; k < N ; k++)
		if (genrand_real1() < .5)
			x[k] = '0';
		else
			x[k] = '1';
	x[N] = '\0';
}

//binary symmetric Markov source startting at state 0
void bsms(int N, char *x, double gamma)
{
	int k, state = 0;
	for (k = 0 ; k < N ; k++) {
		if (genrand_real1() > gamma) state = (state + 1)%2;
		x[k] = '0' + state;
	}
	x[N] = '\0';
}

void set_u_correct_end(char *u, unsigned long p1, unsigned long p2, int Ni, int rec, int mem, char * x) {
	int k;
	unsigned long state = 0UL;
	if (rec == 0) for (k = Ni - mem; k < Ni; k++) u[k] = '0';
	else {
		state = cc(u, p1, p2, x, rec);
		for (k = 0; k < mem; k++) {
			state <<= 1;
			if ((hamming_weight( p1 & state) & 1UL) == 0) u[Ni - mem + k] = '0';
			else u[Ni - mem + k] = '1';
		}
	}

}

double av_dist(double *L_1,double *L_2,int length) {
	int k;
	double dist = 0;
	for (k =0;k<length;k++) dist += (L_1[k] - L_2[k]) * (L_1[k] - L_2[k]);
	dist = dist/length;
	return(dist);
}

/* *****************************************************************************************
   FUNCTIONS TO MEASURE CAPACITY
   ***************************************************************************************** */

double invjfun(double I) {
	double aS1 = 1.09542;
	double bS1 = 0.214217;
	double cS1 = 2.33727;
	double aS2 = 0.706692;
	double bS2 = 0.386013;
	double cS2 = -1.75017;
	double Istar = 0.3646;
	double sigma;
	if (I < Istar) sigma = aS1 * pow(I, 2) + bS1 * I + cS1 * sqrt(I);
	else sigma = - aS2 * log(bS2 * (1 - I)) - cS2 * I;
	return (sigma);
}

double measure_I(int Ni, int mem, double *L_E, char *u) {
	int k;
	double sum = 0, I_out;
	for (k = 0; k < Ni - mem; k++) {
		if (u[k] == '0') {
			sum += log2(1.0 + exp(-L_E[k]));
		}
		else {
			sum += log2(1.0 + exp(+L_E[k]));
		}
	}
	I_out = 1 - (sum / (Ni - mem));
	return (I_out);
}
/* *****************************************************************************************
   MAIN SIMULATION FUNCTIONS
   ***************************************************************************************** */

int measure_I_one_point(int argc, char *argv[]) {
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], d[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2];
	double L_P[MAX_TRANSMIT_LENGTH], L_E[MAX_TRANSMIT_LENGTH], L_in[MAX_TRANSMIT_LENGTH];
	transducer *t, *tch, *t_source;
	trellis *trel, *trel_x;
	unsigned long seed, p1, p2, err, block, dblock;
	int k, Ni, mem, maxdelta, dout;
	double pi, ps, pd, iRate = 1.0, gamma;
	double log_py_and_state0, log_py_and_state1,log_py; 
	double log_py_and_px_and_state0,log_py_and_px_and_state1,log_py_and_px;
	double log_px_and_state0,log_px_and_state1,log_px,log_py_cond;
	FILE *f;
	time_t start;
	unsigned long passes, totpasses = 0UL;
	long double sum = 0.0, sum_cond = 0.0;

	if (argc != 9) {
		printf("Usage: %s 1 Ni md pi pd ps gamma seed\n", argv[0]);
		exit(-1);
	}
	sscanf(argv[2], "%d", &Ni);
	sscanf(argv[3], "%d", &maxdelta);
	sscanf(argv[4], "%lf", &pi);
	sscanf(argv[5], "%lf", &pd);
	sscanf(argv[6], "%lf", &ps);
	sscanf(argv[7], "%lf", &gamma);
	sscanf(argv[8], "%lu", &seed);


	init_genrand(seed);

	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	t_source = Markov_source_transducer(gamma);
	fprint_transducer(t_source, "transducer_source.txt");

	tch = insdel2transducer(pi,pd,ps);
	fprint_transducer(tch, "transducer_channel.txt");
	t = compose_transducers(t_source, tch);
	fprint_transducer(t, "transducer.txt");

	trel = band_trellis(t, Ni, maxdelta, iRate);
	trel_x = band_trellis(t_source, Ni, 0, iRate);
	bcjr_prep(trel);
	bcjr_prep(trel_x);

	time(&start);
	totpasses = 0UL;
	for (block = 0; (block < ITERATIONS_MEASURE_CAPACITY) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bsms(Ni, u, gamma);
		insdelch(u, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);

		if (gamma != 0.5) {
			bcjr(trel, y, u, 0UL, 0UL, Ni, NULL, &totpasses, 1, &log_py_and_state0);
			bcjr(trel, y, u, 0UL, 1UL, Ni, NULL, &totpasses, 1, &log_py_and_state1);
			log_py = max_star(log_py_and_state0, log_py_and_state1);

			bcjr(trel, y, u, 0UL, 0UL, Ni, NULL, &totpasses, 0, &log_py_and_px_and_state0);
			bcjr(trel, y, u, 0UL, 1UL, Ni, NULL, &totpasses, 0, &log_py_and_px_and_state1);
			log_py_and_px = max_star(log_py_and_px_and_state0, log_py_and_px_and_state1);

			bcjr(trel_x, u, u, 0UL, 0UL, Ni, NULL, &totpasses, 1, &log_px_and_state0);
			bcjr(trel_x, u, u, 0UL, 1UL, Ni, NULL, &totpasses, 1, &log_px_and_state1);
			log_px = max_star(log_px_and_state0, log_px_and_state1);
		}
		else {
			bcjr(trel, y, u, 0UL, 0UL, Ni, NULL, &totpasses, 1, &log_py);
			bcjr(trel, y, u, 0UL, 0UL, Ni, NULL, &totpasses, 0, &log_py_and_px);
			bcjr(trel_x, u, u, 0UL, 0UL, Ni, NULL, &totpasses, 1, &log_px);

		}
		log_py /= log(2.0);
		log_py_and_px /= log(2.0);
		log_px /=  log(2.0);

		log_py_cond = log_py_and_px - log_px;

		sum -= log_py;
		sum_cond -= log_py_cond;
	}
	sum /= (Ni * block);
	sum_cond /= (Ni * block);
	printf("%Lf\n", sum - sum_cond);

	free_bcjr(trel);
	free_trellis(trel);
	free_transducer(tch);
	free_transducer(t_source);
	free_transducer(t);
	
	return (0);

}

int oneshot_viterbi(int argc, char *argv[])
{
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], d[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2];
	transducer *t, *tcc, *tch, *tcc1, *t_int, *t_source;
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock;
	int k, Ni, mem, maxdelta, dout, bit_errors, rec;
	double pi, ps, pd, iRate = 2.0;
	unsigned long passes;
	int counti, countd, counts, countt;

	if (argc != 10) {
		printf("Usage: %s 2 p1 p2 rec Ni md pi pd ps\n", argv[0]);
		exit(-1);
	}
	sscanf(argv[2], "%lo", &p1);
	sscanf(argv[3], "%lo", &p2);
	sscanf(argv[4], "%d", &rec);
	sscanf(argv[5], "%d", &Ni);
	sscanf(argv[6], "%d", &maxdelta);
	sscanf(argv[7], "%lf", &pi);
	sscanf(argv[8], "%lf", &pd);
	sscanf(argv[9], "%lf", &ps);

	seed = (unsigned long)time(NULL);
	init_genrand(seed);

	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	if (p2 == 0) iRate = 1.0;

	t_source = Markov_source_transducer(0.5);
	conv_convert(p1, p2, &p1, &p2);
	tcc = conv2transducer(p1, p2, rec);
	tcc1 = timeout_transducer(tcc);
	t_int = compose_transducers(t_source, tcc1);
	tch = insdel2transducer(pi,pd,ps);
	t = compose_transducers(t_int, tch);
	trel = band_trellis(t, Ni, maxdelta, iRate);
	viterbi_prep(trel);

	mem = log2int(p1) > log2int(p2) ? log2int(p1) : log2int(p2);
	mem--;
	bss(Ni, u);

	cc(u, p1, p2, x, rec);
	set_u_correct_end(u, p1, p2, Ni, rec, mem, x);
	cc(u, p1, p2, x, rec);
	insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
	dout = viterbi(trel, y, 0UL, 0UL, Ni, d, &passes);
	bit_errors = count_number_bit_errors (u, d, Ni);
	printf("u: %s\nx: %s\ny: %s\nd: %s\n", u, x, y, d);
	printf("Dout: %d, Passes: %ld\n", dout, passes);

	for (k = 0 , counti = 0, countd = 0, counts = 0, countt = 0; k < strlen(trace) ; k++)
		switch (trace[k]) {
		case 'i':
			counti++;
			break;
		case 'd':
			countd++;
			break;
		case 's':
			counts++;
			break;
		case 't':
			countt++;
			break;
		default:
			printf("Unexpected symbol in trace sequence!\n");
		}

	printf("Insdel channel had %d insertions, %d deletions, %d substitutions and %d correct transmissions\n", counti, countd, counts, countt);
	switch (dout) {
	case -1:
		printf("Viterbi failed to decode because length of channel sequence is out of banded range\n");
		break;
	case -2:
		printf("Viterbi failed to decode because end state was not reached\n");
		break;
	case 1:
		if (strcmp(u, d) != 0)
			printf("Viterbi decoded to a wrong word, %d bit are wrong\n", bit_errors);
		else
			printf("Viterbi decoded successfully\n");
		break;
	default:
		printf("Unexpected value of vitout\n");
	}
	free_viterbi(trel);
	free_trellis(trel);
	free_transducer(tch);
	free_transducer(t);
	free_transducer(tcc);
	free_transducer(tcc1);
	free_transducer(t_source);
	free_transducer(t_int);

	return (0);
}

int oneshot_bcjr(int argc, char *argv[])
{
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], d[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2];
	transducer *t, *tcc, *tch, *tcc1, *t_int, *t_source;
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock;
	int k, Ni, mem, maxdelta, dout, bit_errors, rec;
	double pi, ps, pd, iRate = 2.0;
	unsigned long passes;
	int counti, countd, counts, countt;
	double L[MAX_TRANSMIT_LENGTH];

	if (argc != 10) {
		printf("Usage: %s 3 p1 p2 rec Ni md pi pd ps\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%lo", &p1);
	sscanf(argv[3], "%lo", &p2);
	sscanf(argv[4], "%d", &rec);
	sscanf(argv[5], "%d", &Ni);
	sscanf(argv[6], "%d", &maxdelta);
	sscanf(argv[7], "%lf", &pi);
	sscanf(argv[8], "%lf", &pd);
	sscanf(argv[9], "%lf", &ps);


	seed = (unsigned long)time(NULL);
	init_genrand(seed);

	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}

	if (p2 == 0) iRate = 1.0;

	t_source = Markov_source_transducer(0.5);
	conv_convert(p1, p2, &p1, &p2);
	tcc = conv2transducer(p1, p2, rec);
	tcc1 = timeout_transducer(tcc);
	t_int = compose_transducers(t_source, tcc1);
	tch = insdel2transducer(pi,pd,ps);
	t = compose_transducers(t_int, tch);
	trel = band_trellis(t, Ni, maxdelta, iRate);

	bcjr_prep(trel);
	mem = log2int(p1) > log2int(p2) ? log2int(p1) : log2int(p2);
	mem--;

	bss(Ni, u);
	set_u_correct_end(u, p1, p2, Ni, rec, mem, x);
	cc(u, p1, p2, x, rec);
	insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
	dout = bcjr(trel, y, y, 0UL, 0UL, Ni, NULL, &passes, 1, NULL);
	bcjr_inputprobs(trel, L);
	hard_decision(L, d, 0, Ni);
	bit_errors = count_number_bit_errors (u, d, Ni);
	printf("u: %s\nx: %s\ny: %s\nd: %s\n", u, x, y, d);
	printf("L:");
	for (k = 0 ; k < Ni ; k++)
		printf(" %lf", L[k]);
	printf("\n");
	printf("Dout: %d, Passes: %ld\n", dout, passes);

	for (k = 0 , counti = 0, countd = 0, counts = 0, countt = 0; k < strlen(trace) ; k++)
		switch (trace[k]) {
		case 'i':
			counti++;
			break;
		case 'd':
			countd++;
			break;
		case 's':
			counts++;
			break;
		case 't':
			countt++;
			break;
		default:
			printf("Unexpected symbol in trace sequence!\n");
		}

	printf("Insdel channel had %d insertions, %d deletions, %d substitutions and %d correct transmissions\n", counti, countd, counts, countt);

	switch (dout) {
	case -1:
		printf("BCJR failed to decode because length of channel sequence is out of banded range\n");
		break;
	case 1:
		if (strcmp(u, d) != 0)
			printf("BCJR decoded to a wrong information sequence  %d bits are wrong\n", bit_errors);
		else
			printf("BCJR decoded successfully\n");
		break;
	default:
		printf("Unexpected value of dout\n");
	}

	free_bcjr(trel);
	free_trellis(trel);
	free_transducer(tch);
	free_transducer(t);
	free_transducer(tcc);
	free_transducer(tcc1);
	free_transducer(t_int);
	free_transducer(t_source);
	return (0);
}


int oneshot_markers(int argc, char *argv[])
{
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], dec[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2];
	double Lin[MAX_TRANSMIT_LENGTH], LinM[MAX_TRANSMIT_LENGTH], LoutM[MAX_TRANSMIT_LENGTH], Lout[MAX_TRANSMIT_LENGTH];
	trellis *trel;
	unsigned long seed, err, block, dblock;
	int d, k, Ni, No, mem, maxdelta, dout;
	double pi, ps, pd;
	unsigned long passes;
	int counti, countd, counts, countt;

	if (argc != 8) {
		printf("Usage: %s 5 d Ni md pi pd ps\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%d", &d);
	sscanf(argv[3], "%d", &Ni);
	sscanf(argv[4], "%d", &maxdelta);
	sscanf(argv[5], "%lf", &pi);
	sscanf(argv[6], "%lf", &pd);
	sscanf(argv[7], "%lf", &ps);

	seed = (unsigned long)time(NULL);
	init_genrand(seed);

	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}

	No = Ni + (Ni / d ) * strlen(markers) / 2;
	printf("%d\n", No);
	trel = marker_trellis(d, No, maxdelta,pi,pd,ps,0);
	bcjr_prep(trel);

	printf("Trellis: nsund=%lu, ntund=%lu, nni=%d, md=%d\n", trel->nsund, trel->ntund, trel->nni, trel->maxdelta);
	printf("av_no =");
	for (k = 0 ; k < trel->nni ; k++)
		printf(" %d", trel->av_no[k]);
	printf("\n");
	printf("td->ns = %lu, td->nt = %lu\n", trel->td->ns, trel->td->nt);

	bss(Ni, u);
	marker_encode(u, x, d);
	insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
	for (k = 0 ; k < Ni ; k++)
		Lin[k] = 0.0;
	insert_marker_LLR(Lin, LinM, Ni, d);

	dout = bcjr(trel, y, y, 0UL, 0UL, No, LinM, &passes, 1, NULL);
	bcjr_inputprobs(trel, LoutM);
	remove_marker_LLR(LoutM, Lout, Ni, d);
	hard_decision(Lout, dec, 0, Ni);
	printf("u: %s\nx: %s\ny: %s\nd: %s\n", u, x, y, dec);
	printf("LinM:");
	for (k = 0 ; k < strlen(y) ; k++)
		printf(" %lf", LinM[k]);
	printf("\n");
	printf("LoutM:");
	for (k = 0 ; k < strlen(y) ; k++)
		printf(" %lf", LoutM[k]);
	printf("\n");
	printf("Lout:");
	for (k = 0 ; k < Ni ; k++)
		printf(" %lf", Lout[k]);
	printf("\n");
	printf("Dout: %d, Passes: %ld\n", dout, passes);

	for (k = 0 , counti = 0, countd = 0, counts = 0, countt = 0; k < strlen(trace) ; k++)
		switch (trace[k]) {
		case 'i':
			counti++;
			break;
		case 'd':
			countd++;
			break;
		case 's':
			counts++;
			break;
		case 't':
			countt++;
			break;
		default:
			printf("Unexpected symbol in trace sequence!\n");
		}

	printf("Insdel channel had %d insertions, %d deletions, %d substitutions and %d correct transmissions\n", counti, countd, counts, countt);

	switch (dout) {
	case -1:
		printf("BCJR failed to decode because length of channel sequence is out of banded range\n");
		break;
	case 1:
		if (strcmp(u, dec) != 0)
			printf("BCJR decoded to a wrong information sequence, %d errors\n", count_number_bit_errors(u, dec, Ni));
		else
			printf("BCJR decoded successfully\n");
		break;
	default:
		printf("Unexpected value of dout\n");
	}

	free_bcjr(trel);
	free_trellis(trel);

	return (0);
}

int oneshot_LDPC(int argc, char *argv[]) {
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], dec[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2];
	int k;
	unsigned long seed;
	//Tanner_graph graph;
	Tanner_graph *graph;
	//char message[MAX_TRANSMIT_LENGTH], code[MAX_TRANSMIT_LENGTH];
	double LLR[MAX_TRANSMIT_LENGTH], LLR_2[MAX_TRANSMIT_LENGTH], ps;
	sparse_matrix ** matrixes;

	if (argc != 4) {
		printf("Usage: %s 5 ps file_name_matrixes\n", argv[0]);
		exit(-1);
	}
	sscanf(argv[2], "%lf", &ps);
	matrixes = load_2_matrixes(argv[3]);
	seed = (unsigned long)time(NULL);
	init_genrand(seed);
	bss(matrixes[0]->M, u);
	LDPC_encode_from_gen(u, matrixes[0], x);
	insdelch(x, 0, 0, ps, y, MAX_TRANSMIT_LENGTH, trace);
	//print_matrix(matrixes[0], 1);
	//print_matrix(matrixes[1], 0);
	BSC_LLR(y, ps, LLR);
	BSC_LLR(y, ps, LLR_2);

	LDPC_decoder (matrixes[1], LLR);
	hard_decision(LLR, dec, matrixes[1]->N - matrixes[0]->M, matrixes[1]->N);
	printf("u: %s\n", u);
	printf("x: %s\n", x);
	printf("y: %s\n", y);
	printf("dec: %s\n", dec);
	printf("number of errors:%d\n", count_number_bit_errors (u, dec, matrixes[1]->N - matrixes[1]->M));
	free_sparse_matrix(matrixes[0]);
	free_sparse_matrix(matrixes[1]);
	free(matrixes);
	return (0);
}

int oneshot_bcjr_LDPC(int argc, char *argv[])
{
	char u[MAX_TRANSMIT_LENGTH], x_1[MAX_TRANSMIT_LENGTH], x_2[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], d[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2], file_name_matrixes[1000], outfile_LLR[1000];
	transducer *t, *tcc, *tch, *tcc1, *t_int, *t_source;
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock, num_LDPC_it = 0;
	int k, j, mem, maxdelta, dout, bit_errors, rec;
	double pi, ps, pd, iRate = 2.0, sum, I_out = 0;
	unsigned long passes;
	int counti, countd, counts, countt;
	double L[MAX_TRANSMIT_LENGTH], L2[MAX_TRANSMIT_LENGTH], L_old[MAX_TRANSMIT_LENGTH];;
	sparse_matrix ** matrixes;
	time_t start;
	FILE *f;

	if (argc < 10||argc > 11) {
		printf("Usage: %s 6 p1 p2 rec md pi pd ps file_name_matrixes outfile_LLR\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%lo", &p1);
	sscanf(argv[3], "%lo", &p2);
	sscanf(argv[4], "%d", &rec);
	sscanf(argv[5], "%d", &maxdelta);
	sscanf(argv[6], "%lf", &pi);
	sscanf(argv[7], "%lf", &pd);
	sscanf(argv[8], "%lf", &ps);
	sscanf(argv[9], "%s", file_name_matrixes);
	if (argc == 11) sscanf(argv[10], "%s", outfile_LLR);

	seed = (unsigned long)time(NULL);
	init_genrand(seed);
	matrixes = load_2_matrixes(file_name_matrixes);

	if (3 * matrixes[0]->N > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3*matrixes[0]->N exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	if (p2 == 0) iRate = 1.0;
	mem = log2int(p1) > log2int(p2) ? log2int(p1) : log2int(p2);
	mem--;

	//t_source = Markov_source_transducer(0.5);
	conv_convert(p1, p2, &p1, &p2);
	tcc = conv2transducer(p1, p2, rec);
	tcc1 = timeout_transducer(tcc);
	//t_int = compose_transducers(t_source, tcc1);
	tch = insdel2transducer(pi,pd,ps);
	t = compose_transducers(tcc1, tch);
	trel = band_trellis(t, (matrixes[0]->N) + mem, maxdelta, iRate);

	bcjr_prep(trel);
	bss(matrixes[0]->M, u);
	LDPC_encode_from_gen(u, matrixes[0], x_1);

	for (k = 0; k < mem; k++) {
		x_1[matrixes[0]->N + k] = '0';
	}
	x_1[matrixes[0]->N + k] = '\0';

	set_u_correct_end(x_1, p1, p2, matrixes[0]->N + mem, rec, mem, x_2);
	cc(x_1, p1, p2, x_2, rec);
	insdelch(x_2, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);

	if (argc == 11) assert((f = fopen(outfile_LLR, "a")) != NULL);
	for (k = 0 ; k < matrixes[0]->N + mem; k++) {
		L[k] = 0.0;
		L2[k] = 0.0;
		L_old[k] = 0.0;
	}
	for (j = 0; j < MAX_OUT_IT_TURBO; j++) {
		for (k = 0 ; k < matrixes[0]->N; k++) {
			L[k] -= L2[k];	
			if (argc == 11) fprintf(f, "%d,0,%c,%lf\n", j,x_1[k],L[k]);
			//if (L[k] > MAX_L) L[k] = MAX_L;
			//if (L[k] < -MAX_L) L[k] = -MAX_L;
		}
		if (check_if_inf (L, matrixes[0]->N)) break;
		printf("mutual information out of the LDPC decoder %lf\n", measure_I( matrixes[1]->N, mem, L, x_1));
		dout = bcjr(trel, y, y, 0UL, 0UL, matrixes[0]->N + mem, L, &passes, 1, NULL);
		bcjr_inputprobs(trel, L2);
		if (check_if_inf (L2, matrixes[0]->N)) {
			for (k = 0 ; k < matrixes[0]->N; k++) {
				L[k] = L2[k];
			}
			break;
		}
		for (k = 0 ; k < matrixes[0]->N; k++) {
			L2[k] -= L[k];
			//if (L2[k] > MAX_L) L2[k] = MAX_L;
			//if (L2[k] < -MAX_L) L2[k] = -MAX_L;
			L[k] = L2[k];
			if (argc == 11) fprintf(f, "%d,0,%c,%lf\n", j,x_1[k],L[k]);
		}
		printf("mutual information out of the BCJR decoder %lf\n", measure_I( matrixes[1]->N, mem, L2, x_1));
		num_LDPC_it += LDPC_decoder(matrixes[1], L);
		//if (av_dist(L,L_old,matrixes[0]->N)<EPSILON || check_if_inf (L, matrixes[0]->N) ) j = MAX_OUT_IT_TURBO;
		//printf("distance : %lf\n", av_dist(L,L_old,matrixes[0]->N));
		for (k = 0 ; k < matrixes[0]->N; k++) L_old[k] = L[k];
	}
	if (argc == 11) fclose(f);

	hard_decision(L, d, matrixes[1]->N - matrixes[0]->M, matrixes[1]->N);
	bit_errors = count_number_bit_errors (u, d, matrixes[0]->N);
	printf("u: %s\nx_1: %s\nx_2: %s\ny: %s\nd: %s\n", u, x_1, x_2, y, d);
	printf("L:");
	for (k = 0 ; k < matrixes[0]->N ; k++)
		printf(" %lf", L[k]);
	printf("\n");
	printf("Dout: %d, Passes: %ld\n", dout, passes);

	for (k = 0 , counti = 0, countd = 0, counts = 0, countt = 0; k < strlen(trace) ; k++)
		switch (trace[k]) {
		case 'i':
			counti++;
			break;
		case 'd':
			countd++;
			break;
		case 's':
			counts++;
			break;
		case 't':
			countt++;
			break;
		default:
			printf("Unexpected symbol in trace sequence!\n");
		}

	printf("Insdel channel had %d insertions, %d deletions, %d substitutions and %d correct transmissions\n", counti, countd, counts, countt);

	switch (dout) {
	case -1:
		printf("BCJR failed to decode because length of channel sequence is out of banded range\n");
		break;
	case 1:
		if (strcmp(u, d) != 0)
			printf("BCJR decoded to a wrong information sequence  %d bits are wrong\n", bit_errors);
		else
			printf("BCJR decoded successfully\n");
		break;
	default:
		printf("Unexpected value of dout\n");
	}

	free_bcjr(trel);
	free_trellis(trel);
	free_transducer(tch);
	free_transducer(t);
	free_transducer(tcc);
	free_transducer(tcc1);
	//free_transducer(t_source);
	//free_transducer(t_int);
	free_sparse_matrix(matrixes[0]);
	free_sparse_matrix(matrixes[1]);
	free(matrixes);
	return (0);
}

int oneshot_Marker_LDPC (int argc, char *argv[]) {
	char u[MAX_TRANSMIT_LENGTH], x_1[MAX_TRANSMIT_LENGTH], x_2[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], dec[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2], file_name_matrixes[1000];
	transducer *t, *tcc, *tch, *tcc1;
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock, num_LDPC_it = 0;
	int k, j, maxdelta, dout, bit_errors, d;
	double pi, ps, pd, I_out =0;
	unsigned long passes;
	int counti, countd, counts, countt,No;
	double L[MAX_TRANSMIT_LENGTH], L2[MAX_TRANSMIT_LENGTH],LinM[MAX_TRANSMIT_LENGTH],LoutM[MAX_TRANSMIT_LENGTH], L_old[MAX_TRANSMIT_LENGTH];
	sparse_matrix ** matrixes;
	time_t start;

	if (argc != 8) {
		printf("Usage: %s 7 d md pi pd ps file_name_matrixes \n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%d", &d);
	sscanf(argv[3], "%d", &maxdelta);
	sscanf(argv[4], "%lf", &pi);
	sscanf(argv[5], "%lf", &pd);
	sscanf(argv[6], "%lf", &ps);
	sscanf(argv[7], "%s", file_name_matrixes);

	seed = (unsigned long)time(NULL);
	init_genrand(seed);
	matrixes = load_2_matrixes(file_name_matrixes);

	if (3 * matrixes[0]->N > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3*matrixes[0]->N exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	No = matrixes[0]->N + (matrixes[0]->N / d) * strlen(markers) / 2;
	trel = marker_trellis(d, No, maxdelta,pi, pd, ps,1);
	bcjr_prep(trel);
	bss(matrixes[0]->M, u);
	LDPC_encode_from_gen(u, matrixes[0], x_1);
	marker_encode(x_1, x_2, d);
	insdelch(x_2, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
	for (k = 0 ; k < matrixes[0]->N ; k++) {
		L[k] = 0.0;
		L2[k] = 0.0;
		L_old[k] = 0;
	}

	for (j = 0; j < MAX_OUT_IT_TURBO; j++) {
		for (k = 0 ; k < matrixes[0]->N; k++) L[k] -= L2[k];
		if (check_if_inf (L, matrixes[0]->N)) break;
		insert_marker_LLR(L, LinM, matrixes[0]->N, d);
		dout = bcjr(trel, y, y, 0UL, 0UL, No, LinM, &passes, 1, NULL);
		bcjr_inputprobs(trel, LoutM);
		remove_marker_LLR(LoutM, L2, matrixes[0]->N, d);
		for (k = 0 ; k < matrixes[0]->N; k++) {
			L2[k] -= L[k];
			L[k] = L2[k];
		}
		if (check_if_inf (L2, matrixes[0]->N)) break;
		num_LDPC_it += LDPC_decoder(matrixes[1], L);
		if (av_dist(L,L_old,matrixes[0]->N)<EPSILON) break;
		for (k = 0 ; k < matrixes[0]->N; k++) L_old[k] = L[k];
	}

	hard_decision(L, dec, matrixes[1]->N - matrixes[0]->M, matrixes[1]->N);
	bit_errors = count_number_bit_errors (u, dec, matrixes[0]->N);
	printf("u: %s\nx_1: %s\nx_2: %s\ny: %s\nd: %s\n", u, x_1, x_2, y, dec);
	printf("number of errors %d\n", bit_errors);

	free_bcjr(trel);
	free_trellis(trel);
	free_sparse_matrix(matrixes[0]);
	free_sparse_matrix(matrixes[1]);
	free(matrixes);

	return(1);
}

int batch_viterbi(int argc, char *argv[])
{
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], d[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2],  outfile[1000];
	transducer *t, *tcc, *tch, *tcc1, *t_int, *t_source;
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock;
	int k, Ni, mem, maxdelta, dout, rec;
	double pi, ps, pd, iRate = 2.0;
	FILE *f;
	time_t start;
	unsigned long passes, totpasses = 0UL;

	if (argc != 12) {
		printf("Usage: %s 7 p1 p2 rec Ni md pi pd ps outfile seed\n", argv[0]);
		exit(-1);
	}
	sscanf(argv[2], "%lo", &p1);
	sscanf(argv[3], "%lo", &p2);
	sscanf(argv[4], "%d", &rec);
	sscanf(argv[5], "%d", &Ni);
	sscanf(argv[6], "%d", &maxdelta);
	sscanf(argv[7], "%lf", &pi);
	sscanf(argv[8], "%lf", &pd);
	sscanf(argv[9], "%lf", &ps);
	sscanf(argv[10], "%s", outfile);
	sscanf(argv[11], "%lu", &seed);

	init_genrand(seed);
	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	if (p2 == 0) iRate = 1.0;
	t_source = Markov_source_transducer(0.5);
	conv_convert(p1, p2, &p1, &p2);
	tcc = conv2transducer(p1, p2, rec);
	tcc1 = timeout_transducer(tcc);
	t_int = compose_transducers(t_source, tcc1);
	tch = insdel2transducer(pi,pd,ps);
	t = compose_transducers(t_int, tch);
	trel = band_trellis(t, Ni, maxdelta, iRate);
	viterbi_prep(trel);

	mem = log2int(p1) > log2int(p2) ? log2int(p1) : log2int(p2);
	mem--;

	time(&start);
	totpasses = 0UL;
	for (err = 0 , block = 0, dblock = 0; (err < MIN_ERRORS || block < MIN_BLOCKS) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bss(Ni, u);
		set_u_correct_end(u, p1, p2, Ni, rec, mem, x);
		cc(u, p1, p2, x, rec);
		insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);

		dout = viterbi(trel, y, 0UL, 0UL, Ni, d, &passes);
		if (dout != 1 || strcmp(u, d) != 0) {
			err++;
		}
		if (dout == 1) {
			dblock++;
			totpasses += passes;
		}
	}
	assert((f = fopen(outfile, "a")) != NULL);
	fprintf(f, "%lu,%lo,%lo,%d,%d,%d,%lf,%lf,%lf,%lu,%lu,%lu,%lu \n", seed, p1, p2, rec, Ni, maxdelta, pi, pd, ps, block, err, dblock, totpasses);
	fclose(f);

	free_viterbi(trel);
	free_trellis(trel);
	free_transducer(tch);
	free_transducer(t);
	free_transducer(tcc);
	free_transducer(tcc1);
	free_transducer(t_source);
	free_transducer(t_int);

	return (0);
}

int batch_bcjr(int argc, char *argv[])
{
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], d[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2],  outfile[1000];
	double L[MAX_TRANSMIT_LENGTH];
	transducer *t, *tcc, *tch, *tcc1, *t_int, *t_source;
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock;
	int k, Ni, mem, maxdelta, dout, rec;
	double pi, ps, pd, iRate = 2.0;
	FILE *f;
	time_t start;
	unsigned long passes, totpasses = 0UL;

	if (argc != 12) {
		printf("Usage: %s 8 p1 p2 rec Ni md pi pd ps outfile seed\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%lo", &p1);
	sscanf(argv[3], "%lo", &p2);
	sscanf(argv[4], "%d", &rec);
	sscanf(argv[5], "%d", &Ni);
	sscanf(argv[6], "%d", &maxdelta);
	sscanf(argv[7], "%lf", &pi);
	sscanf(argv[8], "%lf", &pd);
	sscanf(argv[9], "%lf", &ps);
	sscanf(argv[10], "%s", outfile);
	sscanf(argv[11], "%lu", &seed);

	init_genrand(seed);

	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	if (p2 == 0) iRate = 1.0;
	t_source = Markov_source_transducer(0.5);
	conv_convert(p1, p2, &p1, &p2);
	tcc = conv2transducer(p1, p2, rec);
	tcc1 = timeout_transducer(tcc);
	t_int = compose_transducers(t_source, tcc1);
	tch = insdel2transducer(pi,pd,ps);
	t = compose_transducers(t_int, tch);
	trel = band_trellis(t, Ni, maxdelta, iRate);
	bcjr_prep(trel);

	mem = log2int(p1) > log2int(p2) ? log2int(p1) : log2int(p2);
	mem--;

	time(&start);
	totpasses = 0UL;
	for (err = 0 , block = 0, dblock = 0; (err < MIN_ERRORS || block < MIN_BLOCKS) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bss(Ni, u);
		set_u_correct_end(u, p1, p2, Ni, rec, mem, x);
		cc(u, p1, p2, x, rec);
		insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
		dout = bcjr(trel, y, y, 0UL, 0UL, Ni, NULL, &passes, 1, NULL);
		bcjr_inputprobs(trel, L);
		hard_decision(L, d, 0, Ni);
		if (dout != 1 || strcmp(u, d) != 0)
			err++;

		if (dout == 1) {
			dblock++;
			totpasses += passes;
		}
	}

	assert((f = fopen(outfile, "a")) != NULL);
	fprintf(f, "%lu,%lo,%lo,%d,%d,%d,%lf,%lf,%lf,%lu,%lu,%lu,%lu\n", seed, p1, p2, rec, Ni, maxdelta, pi, pd, ps, block, err, dblock, totpasses);
	fclose(f);

	free_bcjr(trel);
	free_trellis(trel);
	free_transducer(tch);
	free_transducer(t);
	free_transducer(tcc);
	free_transducer(tcc1);
	free_transducer(t_source);
	free_transducer(t_int);

	return (0);
}

int batch_markers(int argc, char *argv[]) {

	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], dec[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2], outfile[1000];
	double Lin[MAX_TRANSMIT_LENGTH], LinM[MAX_TRANSMIT_LENGTH], LoutM[MAX_TRANSMIT_LENGTH], Lout[MAX_TRANSMIT_LENGTH];
	trellis *trel;
	unsigned long seed, err, block, dblock;
	int d, k, Ni, No, mem, maxdelta, dout;
	double pi, ps, pd, iRate = 2.0;
	int counti, countd, counts, countt;
	time_t start;
	unsigned long passes, totpasses = 0UL;
	FILE *f;


	if (argc != 10) {
		printf("Usage: %s 9 d Ni md pi pd ps outfile seed\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%d", &d);
	sscanf(argv[3], "%d", &Ni);
	sscanf(argv[4], "%d", &maxdelta);
	sscanf(argv[5], "%lf", &pi);
	sscanf(argv[6], "%lf", &pd);
	sscanf(argv[7], "%lf", &ps);
	sscanf(argv[8], "%s", outfile);
	sscanf(argv[9], "%lu", &seed);

	//seed = (unsigned long)time(NULL);
	init_genrand(seed);
	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	No = Ni + (Ni / d) * strlen(markers) / 2;
	trel = marker_trellis(d, No, maxdelta,pi,pd,ps,0);
	bcjr_prep(trel);
	time(&start);
	totpasses = 0UL;
	for (err = 0 , block = 0, dblock = 0; (err < MIN_ERRORS || block < MIN_BLOCKS) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bss(Ni, u);
		marker_encode(u, x, d);
		insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
		for (k = 0 ; k < Ni ; k++)
			Lin[k] = 0.0;
		insert_marker_LLR(Lin, LinM, Ni, d);
		dout = bcjr(trel, y, y, 0UL, 0UL, No, LinM, &passes, 1, NULL);
		bcjr_inputprobs(trel, LoutM);
		remove_marker_LLR(LoutM, Lout, Ni, d);
		hard_decision(Lout, dec, 0, Ni);
		if (dout != 1 || strcmp(u, dec) != 0)
			err++;
		if (dout == 1) {
			dblock++;
			totpasses += passes;
		}
	}
	assert((f = fopen(outfile, "a")) != NULL);
	fprintf(f, "%lu,%d,%d,%d,%lf,%lf,%lf,%lu,%lu,%lu,%lu\n", seed, d, Ni, maxdelta, pi, pd, ps, block, err, dblock, totpasses);
	fclose(f);
	free_bcjr(trel);
	free_trellis(trel);
	return (0);
}

int batch_LDPC(int argc, char *argv[]) {
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], dec[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2], file_name_matrixes[1000], outfile[1000];
	int k;
	unsigned long seed, err, bit_err, block, num_it = 0;
	Tanner_graph *graph;
	FILE *f;
	double LLR[MAX_TRANSMIT_LENGTH], ps;
	sparse_matrix ** matrixes;
	time_t start;

	if (argc != 6) {
		printf("Usage: %s ps file_name_matrixes outfile seed\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%lf", &ps);
	sscanf(argv[3], "%s", file_name_matrixes);
	sscanf(argv[4], "%s", outfile);
	sscanf(argv[5], "%lu", &seed);

	//  seed = (unsigned long)time(NULL);
	matrixes = load_2_matrixes(file_name_matrixes);
	init_genrand(seed);
	time(&start);
	for (err = 0 , block = 0, bit_err = 0; (err < MIN_ERRORS || block < MIN_BLOCKS) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bss(matrixes[0]->M, u);
		LDPC_encode_from_gen(u, matrixes[0], x);
		insdelch(x, 0, 0, ps, y, MAX_TRANSMIT_LENGTH, trace);
		BSC_LLR(y, ps, LLR);
		num_it = num_it + LDPC_decoder (matrixes[1], LLR);
		hard_decision(LLR, dec, matrixes[1]->N - matrixes[0]->M, matrixes[1]->N);
		if (strcmp(u, dec) != 0) err++;
		bit_err = bit_err + count_number_bit_errors (u, dec, matrixes[0]->M);
	}
	assert((f = fopen(outfile, "a")) != NULL);
	fprintf(f, "%lu,%d,%f,%lu,%lu,%lu,%lu\n", seed, matrixes[0]->M, ps, block, err, bit_err, num_it);
	fclose(f);
	free_sparse_matrix(matrixes[0]);
	free_sparse_matrix(matrixes[1]);
	free(matrixes);
	return (0);
}

int batch_bcjr_EXIT(int argc, char *argv[])
{
	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], d[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2], outfile[1000],outfile_LLR[1000];
	double L_P[MAX_TRANSMIT_LENGTH], L_E[MAX_TRANSMIT_LENGTH], L_in[MAX_TRANSMIT_LENGTH];
	transducer *t, *tcc, *tch, *tcc1, *t_int, *t_source;
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock;
	int k, Ni, mem, maxdelta, dout, rec;
	double pi, ps, pd, I_in, I_out = 0, std_dev, iRate = 2.0;
	FILE *f, *f_LLR;
	time_t start;
	unsigned long passes, totpasses = 0UL;
	long double sum = 0.0;

	if (argc < 13 || argc > 14) {
		printf("Usage: %s mode p1 p2 rec Ni md pi pd ps outfile seed I_in outfile_LLR\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%lo", &p1);
	sscanf(argv[3], "%lo", &p2);
	sscanf(argv[4], "%d", &rec);
	sscanf(argv[5], "%d", &Ni);
	sscanf(argv[6], "%d", &maxdelta);
	sscanf(argv[7], "%lf", &pi);
	sscanf(argv[8], "%lf", &pd);
	sscanf(argv[9], "%lf", &ps);
	sscanf(argv[10], "%s", outfile);
	sscanf(argv[11], "%lu", &seed);
	sscanf(argv[12], "%lf", &I_in);
	if (argc == 14)	sscanf(argv[13], "%s", outfile_LLR);

	//  seed = (unsigned long)time(NULL);
	init_genrand(seed);

	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}

	if (p2 == 0) iRate = 1.0;

	//t_source = Markov_source_transducer(0.5);
	conv_convert(p1, p2, &p1, &p2);
	tcc = conv2transducer(p1, p2, rec);
	tcc1 = timeout_transducer(tcc);
	//t_int = compose_transducers(t_source, tcc1);
	tch = insdel2transducer(pi,pd,ps);
	t = compose_transducers(tcc1, tch);
	trel = band_trellis(t, Ni, maxdelta, iRate);
	bcjr_prep(trel);

	mem = log2int(p1) > log2int(p2) ? log2int(p1) : log2int(p2);
	mem--;

	std_dev = invjfun(I_in);

	if (argc == 14)	assert((f_LLR = fopen(outfile_LLR, "a")) != NULL);
	time(&start);
	totpasses = 0UL;
	for (block = 0; (block < MAX_PACKETS_EXIT_SIM) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bss(Ni, u);
		set_u_correct_end(u, p1, p2, Ni, rec, mem, x);
		for (k = 0; k < Ni; k++) {
			if (u[k] == '0') L_in[k] = 0.5 * pow(std_dev, 2) + std_dev * genrand_norm();
			else L_in[k] = -0.5 * pow(std_dev, 2) + std_dev * genrand_norm();
		}
		cc(u, p1, p2, x, rec);
		insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);

		dout = bcjr(trel, y, y, 0UL, 0UL, Ni, L_in, &passes, 1, NULL);
		bcjr_inputprobs(trel, L_P);
		for (k = 0; k < Ni - mem; k++) {
			L_E[k] = L_P[k] - L_in[k];
			if (argc == 14) fprintf(f_LLR, "%lu,%c,%lf\n", block, u[k], L_E[k]);
		}
		I_out += measure_I(Ni, mem, L_E, u);
	}
	I_out /= block;
	if (argc == 14)	fclose(f_LLR);
	assert((f = fopen(outfile, "a")) != NULL);
	fprintf(f, "%lu,%lo,%lo,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lu\n", seed, p1, p2, rec, Ni, maxdelta, pi, pd, ps, I_in, I_out, block);
	fclose(f);

	free_bcjr(trel);
	free_trellis(trel);
	free_transducer(tch);
	free_transducer(t);
	free_transducer(tcc);
	free_transducer(tcc1);
	//free_transducer(t_source);
	//free_transducer(t_int);

	return (0);
}

int batch_markers_EXIT(int argc, char *argv[]) {

	char u[MAX_TRANSMIT_LENGTH], x[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], dec[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2], outfile[1000];
	double Lin[MAX_TRANSMIT_LENGTH], LinM[MAX_TRANSMIT_LENGTH], LoutM[MAX_TRANSMIT_LENGTH], Lout[MAX_TRANSMIT_LENGTH],L_E[MAX_TRANSMIT_LENGTH];
	trellis *trel;
	unsigned long seed, err, block, dblock;
	int d, k, Ni, No, mem, maxdelta, dout;
	double pi, ps, pd, iRate = 2.0, I_in,I_out,std_dev;
	int counti, countd, counts, countt;
	time_t start;
	unsigned long passes, totpasses = 0UL;
	FILE *f;


	if (argc != 11) {
		printf("Usage: %s 9 d Ni md pi pd ps outfile seed I_in\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%d", &d);
	sscanf(argv[3], "%d", &Ni);
	sscanf(argv[4], "%d", &maxdelta);
	sscanf(argv[5], "%lf", &pi);
	sscanf(argv[6], "%lf", &pd);
	sscanf(argv[7], "%lf", &ps);
	sscanf(argv[8], "%s", outfile);
	sscanf(argv[9], "%lu", &seed);
	sscanf(argv[10], "%lf", &I_in);


	//seed = (unsigned long)time(NULL);
	init_genrand(seed);
	if (3 * Ni > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3Ni exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}
	No = Ni + (Ni / d) * strlen(markers) / 2;
	trel = marker_trellis(d, No, maxdelta,pi,pd,ps,1);
	bcjr_prep(trel);
	time(&start);
	totpasses = 0UL;
	std_dev = invjfun(I_in);
	for (err = 0 , block = 0, dblock = 0; (err < MIN_ERRORS || block < MIN_BLOCKS) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bss(Ni, u);
		marker_encode(u, x, d);
		insdelch(x, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
		for (k = 0; k < Ni; k++) {
			if (u[k] == '0') Lin[k] = 0.5 * pow(std_dev, 2) + std_dev * genrand_norm();
			else Lin[k] = -0.5 * pow(std_dev, 2) + std_dev * genrand_norm();
		}
		insert_marker_LLR(Lin, LinM, Ni, d);
		dout = bcjr(trel, y, y, 0UL, 0UL, No, LinM, &passes, 1, NULL);
		bcjr_inputprobs(trel, LoutM);
		remove_marker_LLR(LoutM, Lout, Ni, d);
		for (k = 0; k < Ni; k++) {
			L_E[k] = Lout[k] - Lin[k];
		}
		I_out += measure_I(Ni, 0, L_E, u);
		
	}
	I_out /= block;

	assert((f = fopen(outfile, "a")) != NULL);
	fprintf(f, "%lu,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lu\n", seed, d, Ni, maxdelta, pi, pd, ps, I_in, I_out, block);
	fclose(f);

	free_bcjr(trel);
	free_trellis(trel);
	return (0);
}

int batch_markers_LDPC(int argc, char *argv[]) {
	char u[MAX_TRANSMIT_LENGTH], x_1[MAX_TRANSMIT_LENGTH], x_2[MAX_TRANSMIT_LENGTH], y[MAX_TRANSMIT_LENGTH], dec[MAX_TRANSMIT_LENGTH], trace[MAX_TRANSMIT_LENGTH * 2], file_name_matrixes[1000], outfile[1000];
	trellis *trel;
	unsigned long seed, p1, p2, err, block, dblock, num_LDPC_it = 0,passes, totpasses = 0UL;
	int k, j, maxdelta, dout, bit_errors, d,counti, countd, counts, countt,No;
	double pi, ps, pd;
	double L[MAX_TRANSMIT_LENGTH], L2[MAX_TRANSMIT_LENGTH],LinM[MAX_TRANSMIT_LENGTH],LoutM[MAX_TRANSMIT_LENGTH],L_old[MAX_TRANSMIT_LENGTH];
	sparse_matrix ** matrixes;
	time_t start;
	FILE *f;

	if (argc != 10) {
		printf("Usage: %s 16 d maxdelta pi pd ps outfile file_name_matrixes seed\n", argv[0]);
		exit(-1);
	}

	sscanf(argv[2], "%d", &d);
	sscanf(argv[3], "%d", &maxdelta);
	sscanf(argv[4], "%lf", &pi);
	sscanf(argv[5], "%lf", &pd);
	sscanf(argv[6], "%lf", &ps);
	sscanf(argv[7], "%s", outfile);
	sscanf(argv[8], "%s", file_name_matrixes);
	sscanf(argv[9], "%lu", &seed);


	//seed = (unsigned long)time(NULL);

	init_genrand(seed);
	matrixes = load_2_matrixes(file_name_matrixes);
	if (3 * matrixes[0]->N > MAX_TRANSMIT_LENGTH) {
		fprintf(stderr, "3*matrixes[0]->N exceeds MAX_TRANSMIT_LENGTH: increase parameter and recompile\n");
		exit(-1);
	}

	No = matrixes[0]->N + (matrixes[0]->N / d) * strlen(markers) / 2;
	trel = marker_trellis(d, No, maxdelta,pi,pd,ps,1);
	bcjr_prep(trel);
	time(&start);
	totpasses = 0UL;
	for (err = 0 , block = 0; (err < MIN_ERRORS || block < MIN_BLOCKS) && (difftime(time(NULL), start) < MAX_SIM_TIME) ; block++) {
		bss(matrixes[0]->M, u);
		LDPC_encode_from_gen(u, matrixes[0], x_1);
		marker_encode(x_1, x_2, d);
		insdelch(x_2, pi, pd, ps, y, MAX_TRANSMIT_LENGTH, trace);
		for (k = 0 ; k < matrixes[0]->N ; k++) {
			L[k] = 0.0;
			L2[k] = 0.0;
			L_old[k] = 0.0;
		}
		for (j = 0; j < MAX_OUT_IT_TURBO; j++) {
			//printf("block number:%lu, iteration number: %d \n", block,j);
			for (k = 0 ; k < matrixes[0]->N; k++) L[k] -= L2[k];
			if (check_if_inf (L, matrixes[0]->N)) break;
			insert_marker_LLR(L, LinM, matrixes[0]->N, d);
			dout = bcjr(trel, y, y, 0UL, 0UL, No, LinM, &passes, 1, NULL);
			bcjr_inputprobs(trel, LoutM);
			remove_marker_LLR(LoutM, L2, matrixes[0]->N, d);
			for (k = 0 ; k < matrixes[0]->N; k++) {
				L2[k] -= L[k];
				L[k] = L2[k];
			}
			if (check_if_inf (L2, matrixes[0]->N)) break;
			num_LDPC_it += LDPC_decoder(matrixes[1], L);
			if (av_dist(L,L_old,matrixes[0]->N)<EPSILON) break;
			for (k = 0 ; k < matrixes[0]->N; k++) L_old[k] = L[k];
		}
		hard_decision(L, dec, matrixes[1]->N - matrixes[0]->M, matrixes[1]->N);
		if (strcmp(u, dec) != 0) err++;
	}
	assert((f = fopen(outfile, "a")) != NULL);
	fprintf(f, "%d,%d,%lf,%lf,%lf,%lu,%lu\n", d, maxdelta, pi, pd, ps,block, err);
	fclose(f);
	free_bcjr(trel);
	free_trellis(trel);
	free_sparse_matrix(matrixes[0]);
	free_sparse_matrix(matrixes[1]);
	free(matrixes);
	return (0);
}

int main(int argc, char *argv[])
{
	int mode;
	sscanf(argv[1], "%d", &mode);
	switch (mode) {
	case 1:
		return (measure_I_one_point(argc, argv));
		break;
	case 2:
		return (oneshot_viterbi(argc, argv));
		break;
	case 3:
		return (oneshot_bcjr(argc, argv));
		break;
	case 4:
		oneshot_LDPC(argc, argv);
		break;
	case 5:
		return (oneshot_markers(argc, argv));
		break;
	case 6:
		return (oneshot_bcjr_LDPC(argc, argv));
		break;
	case 7:
		return (oneshot_Marker_LDPC(argc, argv));
		break;
	case 8:
		return (batch_viterbi(argc, argv));
		break;
	case 9:
		return (batch_bcjr(argc, argv));
		break;
	case 10:
		return (batch_markers(argc, argv));
		break;
	case 11:
		batch_LDPC(argc, argv);
		break;
	case 12:
		//batch_BCJR_LDPC
		break;
	case 13:
		//batch_BCJR_Marker
		break;
	case 14:
		return (batch_bcjr_EXIT(argc, argv));
		break;
	case 15:
		return (batch_markers_EXIT(argc, argv));
		break;
	case 16:
		return (batch_markers_LDPC(argc, argv));
		break;
	}
}

/* *********************************************************************************************
 * Random number generators                                                                    *
 ********************************************************************************************* */


/* THE FIRST PART OF THIS FILE IS IMPORTED: it's a relatively powerful
   random number generator using a method called "Mersenne Twister".
   Shift to the "main" routine at the end for our code  */

/*
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

/* initializes mt[N] with a seed */


/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[N]; /* the array for the state vector  */
static int mti = N + 1; /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
	mt[0] = s & 0xffffffffUL;
	for (mti = 1; mti < N; mti++) {
		mt[mti] =
		    (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
		/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
		/* In the previous versions, MSBs of the seed affect   */
		/* only MSBs of the array mt[].                        */
		/* 2002/01/09 modified by Makoto Matsumoto             */
		mt[mti] &= 0xffffffffUL;
		/* for >32 bit machines */
	}
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned long init_key[], int key_length)
{
	int i, j, k;
	init_genrand(19650218UL);
	i = 1; j = 0;
	k = (N > key_length ? N : key_length);
	for (; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL))
		        + init_key[j] + j; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++; j++;
		if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
		if (j >= key_length) j = 0;
	}
	for (k = N - 1; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941UL))
		        - i; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++;
		if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
	}

	mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
	unsigned long y;
	static unsigned long mag01[2] = {0x0UL, MATRIX_A};
	/* mag01[x] = x * MATRIX_A  for x=0,1 */

	if (mti >= N) { /* generate N words at one time */
		int kk;

		if (mti == N + 1) /* if init_genrand() has not been called, */
			init_genrand(5489UL); /* a default initial seed is used */

		for (kk = 0; kk < N - M; kk++) {
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		for (; kk < N - 1; kk++) {
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
		mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

		mti = 0;
	}

	y = mt[mti++];

	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return y;
}

/* generates a random number on [0,0x7fffffff]-interval */
long genrand_int31(void)
{
	return (long)(genrand_int32() >> 1);
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(void)
{
	return genrand_int32() * (1.0 / 4294967295.0);
	/* divided by 2^32-1 */
}

/* generates a sample from a mean 1 exponential distribution */
double genrand_exp1(void) {
	return -log(1 - genrand_real1());
}

/* generates a sample from a 0 mean normal distribution with variance 1*/
double genrand_norm(void)
{
	double b = (genrand_real1() - 0.5) * M_PI;
	double a = genrand_exp1();

	return sqrt(2 * a) * sin(b);
	/* divided by 2^32-1 */
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void)
{
	return genrand_int32() * (1.0 / 4294967296.0);
	/* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double genrand_real3(void)
{
	return (((double)genrand_int32()) + 0.5) * (1.0 / 4294967296.0);
	/* divided by 2^32 */
}

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(void)
{
	unsigned long a = genrand_int32() >> 5, b = genrand_int32() >> 6;
	return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

