#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include <mpi.h>
#include <set>
#include <map>
int liblinear_version = LIBLINEAR_VERSION;
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class T> static inline void swap(T* a, int x, int y) { T t=a[x]; a[x]=a[y]; a[y]=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void print_null(const char *s) {}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

double static inline ddot3(const double *x, const double *y, const double *z, int l)
{
#if 0
	double s = 0;
	for(int i=0; i<l; i++)
		s += (x[i] * y[i] * z[i]);
	return s;
#else
	double s = 0;
	int m = l-4;
	int i;
	for (i = 0; i < m; i += 5)
		s +=    x[i+0] * y[i+0] * z[i+0] +
			x[i+1] * y[i+1] * z[i+1] +
			x[i+2] * y[i+2] * z[i+2] +
			x[i+3] * y[i+3] * z[i+3] +
			x[i+4] * y[i+4] * z[i+4] ;

	for ( ; i < l; i++)        /* clean-up loop */
		s += x[i] * y[i] * z[i];

	return s;
#endif
}

#define MEPS 1e-16L
#define PIVOTING
#define DEBUG_LDL

//This subroutine solves a linear system with a symmetric matrix using
//the LDLT decomposition with pivoting The implementation follows
//Section 4.4 in "Matrix Computation" 4th edition by Gene H. Golub and
//Charles F. Van Loan
double solve_system(const double *_A, const double *_b, int n, int m, double *x)
{
	long double *A = new long double[n * n];
	for(int i=0; i<n*n; i++)
		A[i] = _A[i];
	long double *b = new long double[n];
	for(int i=0; i<n; i++)
		b[i] = _b[i];

#ifdef PIVOTING
	int *p = new int[m];
	for(int i=0; i<m; i++)
		p[i] = i;
#endif

	// decomp
	// A -> D0
	//      L00 D1
	//      L10 L11 D2
	for(int j=0; j<m; j++){
#ifdef PIVOTING
		int pj = j;
		long double pj_val = fabsl(A[j*n+j]);
		// find piv
		for(int i=j+1; i<m; i++){
			long double pi_val = fabsl(A[i*n+i]);
			if(pj_val < pi_val){
				pj_val = pi_val;
				pj = i;
			}
		}
		// swap rows and columns of j and pj
		for(int i=0; i<m; i++)
			swap(A, j*n+i, pj*n+i);
		for(int i=0; i<m; i++)
			swap(A, i*n+j, i*n+pj);
		swap(b, j, pj);
		swap(p, j, pj);
#endif

		// loop invariant: Aik = Lik, Akk = Dk with k<j
		long double Dj = A[j*n+j];
		for(int k=0; k<j; k++){
			Dj -= A[j*n+k]*A[j*n+k] * A[k*n+k];
		}
		A[j*n+j] = Dj;
		//assert(fabs(Dj) >= MEPS);
		if(fabsl(Dj) < MEPS){
			for(int i=j+1; i<m; i++)
				A[i*n+j] = 0;
			continue; // xj should be 0
		}

		for(int i=j+1; i<m; i++){
			long double Lij = A[i*n+j];
			for(int k=0; k<j; k++){
				Lij -= A[i*n+k] * A[j*n+k] * A[k*n+k];
			}
			Lij /= Dj;
			A[i*n+j] = Lij;
		}
	}
#if 0
	printf("L\n");
	for(int i=0; i<n; i++){
		for(int j=0; j<=i; j++)
			printf("%+.3Le ", A[i*n+j]);
		printf("\n");
	}
#endif
	// invariant: A[i*n+j] = Lij, forall i>j
	//            A[i*n+i] = Dii, forall i

	// LD y = b
	// di yi =  bi - \sum_{j<i} yj Lij
	// place y in b
	for(int i=0; i<m; i++){
		// loop invariant: bk = yk forall k<i
		long double yi = b[i];
		for(int k=0; k<i; k++)
			yi -= A[i*n+k] * b[k];
		b[i] = yi;
	}
	for(int i=0; i<m; i++){
		long double Di = A[i*n+i];
		if(fabsl(Di) < MEPS)
			b[i] = 0;
		else
			b[i] /= Di;
	}
#if 0
	printf("y\n");
	for(int i=0; i<n; i++){
		printf("%+.3e\n", b[i]);
	}
#endif
	// L'x = y
	// xi = yi - \sum{j>i} xj Lij
	for(int i=m-1; i>=0; i--){
		// loop invariant: xk ready forall k>i
		long double xi = b[i];
#ifdef PIVOTING
		for(int k=m-1; k>i; k--)
			xi -= A[k*n+i] * x[p[k]];
		x[p[i]] = (double)xi;
#else
		for(int k=m-1; k>i; k--)
			xi -= A[k*n+i] * x[k];
		x[i] = (double)xi;
#endif
	}
#if 0
	printf("\n");
	for(int i=0; i<n; i++)
		printf("d %.3e\n", x[i]);

	printf("A\n");
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++)
			printf("%+.3e ", Acopy[i*n+j]);
		printf("\n");
	}
	printf("b\n");
	for(int i=0; i<n; i++){
		printf("%+.3e\n", bcopy[i]);
	}
#endif

#if 1
	double err = 0;
	for(int i=0; i<m; i++){
		double Ax_i = 0;
		for(int j=0; j<m; j++)
			Ax_i += _A[i*n+j] * x[j];
		err += fabs(Ax_i-_b[i]);
	}
	double max_pv = 0;
	for(int i=0; i<m; i++)
		if(max_pv < _A[i*n+i])
			max_pv = (double)_A[i*n+i];
	if(max_pv != 0)
		err /= max_pv;
	//printf("err=%e\n", err);
	//assert(err < 1e-5);
#endif
	delete[] A;
	delete[] b;
#ifdef PIVOTING
	delete[] p;
#endif
	return err;
}


class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

//This class is an abstract class for L2-regularized, linear, almost twice-differentiable empirical risk minimization problems
class l2r_erm_fun: public function
{
public:
	l2r_erm_fun(const problem *prob, double *C);
	~l2r_erm_fun();

	double fun(double *w);
	double line_search(double *d, double *w, double gTd, double dTd, double alpha, double *f, double *Xd);
	int get_nr_variable(void);

	void Xv(double *v, double *Xv);
	virtual void Hessian_coeff(double *C_times_D) = 0;

protected:
	virtual double C_times_loss(int i, double wx_i) = 0;
	void XTv(double *v, double *XTv);

	double *C;
	const problem *prob;
	double *wx;
	double *tmp;
	double wTw;
	double current_f;
	int start;
	int length;
};

l2r_erm_fun::l2r_erm_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	wx = new double[l];
	tmp = new double[l];
	this->C = C;

	int w_size = get_nr_variable();
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	this->start = shift * rank;
	this->length = min(max(w_size - start, 0), shift);
	if (length == 0)
		start = 0;
}

l2r_erm_fun::~l2r_erm_fun()
{
	delete[] wx;
	delete[] tmp;
}

double l2r_erm_fun::fun(double *w)
{
	int i;
	double f=0;
	double buffer[2];
	int l=prob->l;

	wTw = 0;
	Xv(w, wx);

	for(i=start;i<start + length;i++)
		wTw += w[i] * w[i];
	for(i=0;i<l;i++)
		f += C_times_loss(i, wx[i]);

	buffer[0] = f;
	buffer[1] = wTw;
	mpi_allreduce(buffer, 2, MPI_DOUBLE, MPI_SUM);
	f = buffer[0];
	wTw = buffer[1];
	f = f + 0.5 * wTw;

	current_f = f;
	return(f);
}

int l2r_erm_fun::get_nr_variable(void)
{
	return prob->n;
}

double l2r_erm_fun::line_search(double *d, double *w, double gTd, double dTd, double alpha, double *f, double *Xd)
{
	int i;
	int l = prob->l;
	double wTd = 0;
	double eta = 0.01;
	int max_num_linesearch = 1000;

	memcpy(tmp, Xd, sizeof(double) * l);

	for (i=0;i<length;i++)
		wTd += d[i] * w[start + i];
	mpi_allreduce(&wTd, 1, MPI_DOUBLE, MPI_SUM);
	int num_linesearch = 0;
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		double loss = 0;
		for(i=0;i<l;i++)
		{
			double inner_product = tmp[i] * alpha + wx[i];
			loss += C_times_loss(i, inner_product);
		}
		mpi_allreduce(&loss, 1, MPI_DOUBLE, MPI_SUM);
		*f = loss + (alpha * alpha * dTd + wTw) / 2.0 + alpha * wTd;
		if (*f - current_f <= eta * alpha * gTd)
		{
			for (i=0;i<l;i++)
				wx[i] += alpha * tmp[i];
			break;
		}
		else
			alpha *= 0.5;
	}

	if (num_linesearch >= max_num_linesearch)
	{
		*f = current_f;
		return 0;
	}
	
	wTw += alpha * alpha * dTd + 2* alpha * wTd;
	current_f = *f;
	return alpha;
}

void l2r_erm_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_erm_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
		sparse_operator::axpy(v[i], x[i], XTv);
}

class L_COMMDIR_BFGS
{
public:
	L_COMMDIR_BFGS(const l2r_erm_fun *fun_obj, int l, double eps = 0.1, int m=5, int max_iter = 2000);
	~L_COMMDIR_BFGS();

	void l_commdir_bfgs(double *w);

private:
	l2r_erm_fun *fun_obj;
	void update_inner_products(double** inner_product_matrix, int k, int n, int DynamicM, int previous_M, double** u, double** s, double* g, double* delta, double step_size);
	void compute_system(int iter, int k, int n, int DynamicM, double** inner_product_matrix, double** ux, double** sx, double* gx, double* C_times_D, double* system);
	int l;
	double eps;
	int max_iter;
	int M;
	int start;
	int length;
	int offset;
	int *recv_count;
	int *displace;
};

L_COMMDIR_BFGS::L_COMMDIR_BFGS(const l2r_erm_fun *fun_obj, int l, double eps, int m, int max_iter)
{
	this->fun_obj=const_cast<l2r_erm_fun *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	this->M = m;
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int n = this->fun_obj->get_nr_variable();
	int shift = int(ceil(double(n) / double(nr_node)));
	recv_count = new int[nr_node];
	displace = new int[nr_node];
	start = shift * rank;
	length = min(max(n - start, 0), shift);
	if (length == 0)
		start = 0;
	offset = 0;
	int counter = 0;
	for (int i=0;i<nr_node;i++)
	{
		recv_count[i] = shift;
		displace[i] = counter;
		counter += shift;
		if (counter >= n)
		{
			counter = 0;
			shift = 0;
		}
		else if (counter + shift > n)
			shift = n - counter;
	}
	this->l = l;
}

L_COMMDIR_BFGS::~L_COMMDIR_BFGS()
{
	delete[] recv_count;
	delete[] displace;
}

void L_COMMDIR_BFGS::l_commdir_bfgs(double *w)
{
	int n = fun_obj->get_nr_variable();
	int i, k;
	double f;
	double minusone = -1;
	double one = 1;
	int search = 1, iter = 0, inc = 1;
	int DynamicM = 0;
	double **inner_product_matrix = new double*[2*M+1];
	double *g = new double[n];
	double *Xg = new double[l];
	double *old_g;
	double *Xold_g;
	double *p;
	double *Xp;
	double* t = new double[M * 2 + 1];
	double **Xu = new double*[M];
	double **Xs = new double*[M];
	
	double **u = new double*[M];
	double **s = new double*[M];
	
	for (int i=0; i < 2*M + 1; i++)
		inner_product_matrix[i] = new double[2*M+1];
	for (int i=0; i < M;i++)
		s[i] = new double[length];
	double* PTg = new double[2 * M + 1];
	double *C_times_D = new double[l];
	double gnorm, gnorm0;
	double step_size = 0;
	double gTp = 0;
	double err = 0;
	double pTp;
	double *w0 = new double[n];

	f = fun_obj->fun(w0);
	fun_obj->grad(w0, g);
	gnorm0 = dnrm2_(&n, g, &inc);
	delete [] w0;

	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	fun_obj->Xv(g, Xg);
	gnorm = dnrm2_(&n, g, &inc);

	if (gnorm <= eps*gnorm0)
		search = 0;

	inner_product_matrix[0][0] = gnorm * gnorm;

	int tmp_length = 0;
	int previous_M = 0;
	k = 0;
	double *system = new double[(2 * M + 1) * (2 * M + 1)];

	while (iter <= max_iter && search)
	{
		fun_obj->Hessian_coeff(C_times_D);
		gnorm = sqrt(inner_product_matrix[2 * DynamicM][2 * DynamicM]);
		info("iter %d m %d |g| %5.3e f %5.3e step %5.3e err %5.3e\n", iter,DynamicM, gnorm, f, step_size, err);
		if (gnorm <= eps*gnorm0)
			break;

		memset(system, 0, sizeof(double) * size_t((2 * M + 1) * (2 * M + 1)));
		compute_system(iter, k, n, DynamicM, inner_product_matrix, Xu, Xs, Xg, C_times_D, system);
		
		if (iter == 0)
			k--;
		k = (k+1)%M;

		memset(t, 0, sizeof(double) * size_t(M * 2 + 1));
		tmp_length = 2 * DynamicM + 1;
		memcpy(PTg, inner_product_matrix[2 * DynamicM], sizeof(double) * size_t(tmp_length));
		if (iter <= M)
			for (i=0;i<DynamicM;i++)
				PTg[2 * i] = 0;

		dscal_(&tmp_length, &minusone, PTg, &inc);
		err = solve_system(system, PTg, 2 * DynamicM + 1, tmp_length, t);

		p = new double[length];
		Xp = new double[l];
		memcpy(p, g + start, sizeof(double) * size_t(length));
		memcpy(Xp, Xg, sizeof(double) * size_t(l));
		dscal_(&length, t + (2 * DynamicM), p, &inc);
		dscal_(&l, t + (2 * DynamicM), Xp, &inc);

		for (i=0;i<DynamicM;i++)
		{
			daxpy_(&length, t + (2 * i + 1), s[i], &inc, p, &inc);
			daxpy_(&l, t + (2 * i + 1), Xs[i], &inc, Xp, &inc);
		}

		if (iter > M)
			for (i=0;i<DynamicM;i++)
			{
				daxpy_(&length, t + (2 * i), u[i], &inc, p, &inc);
				daxpy_(&l, t + (2 * i), Xu[i], &inc, Xp, &inc);
			}

		gTp = -ddot_(&tmp_length, t, &inc, PTg, &inc);
		pTp = 0;
		for (i=0;i<2 * DynamicM + 1; i++)
			pTp += t[i] * ddot_(&tmp_length, t, &inc, inner_product_matrix[i], &inc);
		double f_old = f;
		step_size = fun_obj->line_search(p, w, gTp, pTp, 1.0, &f, Xp);
		double actred = f_old - f;
		if (step_size == 0)
		{
			info("WARNING: step size too small\n");
			break;
		}
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred too small\n");
			break;
		}
		dscal_(&length, &step_size, p, &inc);
		dscal_(&l, &step_size, Xp, &inc);
		daxpy_(&length, &one, p, &inc, w + start, &inc);
		old_g = g;
		Xold_g = Xg;
		iter++;
		previous_M = DynamicM;
		DynamicM = min(iter, M);

		if (iter > M)
		{
			delete[] u[k];
			delete[] Xu[k];
			delete[] Xs[k];
		}

		u[k] = p;
		Xu[k] = Xp;
		memcpy(s[k], old_g + start, sizeof(double) * size_t(length));
		Xs[k] = Xold_g;
		delete[] old_g;
		g = new double[n];
		fun_obj->grad(w, g);
		Xg = new double[l];
		fun_obj->Xv(g, Xg);

		daxpy_(&length, &minusone, g + start, &inc, s[k], &inc);
		daxpy_(&l, &minusone, Xg, &inc, Xs[k], &inc);
		update_inner_products(inner_product_matrix, k, n, DynamicM, previous_M, u, s, g, t, step_size);
	}

	delete[] t;
	delete[] system;
	for (i=0;i<DynamicM;i++)
	{
		delete[] Xs[i];
		delete[] s[i];
		delete[] Xu[i];
		delete[] u[i];
	}
	delete[] g;
	delete[] s;
	delete[] u;
	delete[] Xg;
	delete[] Xs;
	delete[] Xu;
	delete[] C_times_D;
	delete[] PTg;

	double *recv_buffer = new double[n];
	MPI_Allgatherv(w + start, length, MPI_DOUBLE, recv_buffer, recv_count, displace, MPI_DOUBLE, MPI_COMM_WORLD);
	memcpy(w, recv_buffer, sizeof(double) * size_t(n));

	delete[] recv_buffer;
}

void L_COMMDIR_BFGS::compute_system(int iter, int k, int n, int DynamicM, double** inner_product_matrix, double** ux, double** sx, double* gx, double* C_times_D, double* system)
{
	int i,j;
	int buffer_length = (2 * DynamicM + 1) * (2 * DynamicM + 2) / 2;
	int counter = 0;

	if (iter <= DynamicM)
		buffer_length = (DynamicM + 2) * (DynamicM + 1) / 2;

	double* buffer = new double[buffer_length];
	double entry;

	if (iter <= DynamicM)
	{
		for (i=0;i<DynamicM;i++)
		{
			for (j=0;j<=i;j++)
			{
				buffer[counter] = ddot3(sx[i], C_times_D, sx[j], l);
				counter++;
			}
		}
		for (i=0;i<DynamicM;i++)
		{
			buffer[counter] = ddot3(gx, C_times_D, sx[i], l);
			counter++;
		}
	}
	else
	{
		for (i=0;i<DynamicM;i++)
		{
			for (j=0;j<i;j++)
			{
				buffer[counter] = ddot3(ux[i], C_times_D, ux[j], l);
				counter++;
				buffer[counter] = ddot3(ux[i], C_times_D, sx[j], l);
				counter++;
			}
			buffer[counter] = ddot3(ux[i], C_times_D, ux[i], l);
			counter++;

			for (j=0;j<=i;j++)
			{
				buffer[counter] = ddot3(sx[i], C_times_D, ux[j], l);
				counter++;
				buffer[counter] = ddot3(sx[i], C_times_D, sx[j], l);
				counter++;
			}
		}
		for (i=0;i<DynamicM;i++)
		{
			buffer[counter] = ddot3(gx, C_times_D, ux[i], l);
			counter++;
			buffer[counter] = ddot3(gx, C_times_D, sx[i], l);
			counter++;
		}
	}
	buffer[counter] = ddot3(gx, C_times_D, gx, l);
	counter++;

	mpi_allreduce(buffer, counter, MPI_DOUBLE, MPI_SUM);
	counter = 0;
	if (iter <= DynamicM)
	{
		for (i=0;i<DynamicM;i++)
			for (j=0;j<=i;j++)
			{
				entry = buffer[counter] + inner_product_matrix[2 * i + 1][2 * j + 1];
				counter++;
				system[(2 * i + 1) * (2 * DynamicM + 1) + 2 * j + 1] = entry;
				system[(2 * j + 1) * (2 * DynamicM + 1) + 2 * i + 1] = entry;
			}
		for (i=0;i<DynamicM;i++)
		{
			memset(system + (2 * i) * (2 * DynamicM + 1), 0, sizeof(double) * size_t(2 * DynamicM + 1));
			for (j=0;j<DynamicM;j++)
				system[(2 * j + 1) * (2 * DynamicM + 1) + 2 * i] = 0;
			system[(2 * DynamicM) * (2 * DynamicM + 1) + 2 * i] = 0;
		}
		for (i=0;i<DynamicM;i++)
		{
			entry = buffer[counter] + inner_product_matrix[2 * DynamicM][2 * i + 1];
			counter++;
			system[(2 * DynamicM) * (2 * DynamicM + 1) + 2 * i + 1] = entry;
			system[(2 * i + 1) * (2 * DynamicM + 1) + 2 * DynamicM] = entry;
		}
	}
	else
	{
		for (i=0;i<DynamicM;i++)
		{
			for (j=0;j<i;j++)
			{
				entry = buffer[counter] + inner_product_matrix[2 * i][2 * j];
				counter++;
				system[2 * i * (2 * DynamicM + 1) + 2 * j] = entry;
				system[2 * j * (2 * DynamicM + 1) + 2 * i] = entry;
				entry = buffer[counter] + inner_product_matrix[2 * i][2 * j + 1];
				counter++;
				system[2 * i * (2 * DynamicM + 1) + 2 * j + 1] = entry;
				system[(2 * j + 1) * (2 * DynamicM + 1) + 2 * i] = entry;
			}
			system[2 * i * (2 * DynamicM + 1) + 2 * i] = buffer[counter] + inner_product_matrix[2 *i][2 * i];
			counter++;

			for (j=0;j<=i;j++)
			{
				entry = buffer[counter] + inner_product_matrix[2 * i + 1][2 * j];
				counter++;
				system[(2 * i + 1) * (2 * DynamicM + 1) + 2 * j] = entry;
				system[(2 * j) * (2 * DynamicM + 1) + 2 * i + 1] = entry;
				entry = buffer[counter] + inner_product_matrix[2 * i + 1][2 * j + 1];
				counter++;
				system[(2 * i + 1) * (2 * DynamicM + 1) + 2 * j + 1] = entry;
				system[(2 * j + 1) * (2 * DynamicM + 1) + 2 * i + 1] = entry;
			}
		}

		for (i=0;i<DynamicM;i++)
		{
			entry = buffer[counter] + inner_product_matrix[2 * DynamicM][2 * i];
			counter++;
			system[(2 * DynamicM) * (2 * DynamicM + 1) + 2 * i] = entry;
			system[(2 * i) * (2 * DynamicM + 1) + 2 * DynamicM] = entry;
			entry = buffer[counter] + inner_product_matrix[2 * DynamicM][2 * i + 1];
			counter++;
			system[(2 * DynamicM) * (2 * DynamicM + 1) + 2 * i + 1] = entry;
			system[(2 * i + 1) * (2 * DynamicM + 1) + 2 * DynamicM] = entry;
		}
	}
	system[(2 * DynamicM + 1) * (2 * DynamicM + 1) - 1] = buffer[counter] + inner_product_matrix[2 * DynamicM][2 * DynamicM];
	
	delete[] buffer;
}

void L_COMMDIR_BFGS::update_inner_products(double** inner_product_matrix, int k, int n, int DynamicM, int previous_M, double** u, double** s, double* g, double* t, double step_size)
{
	int i;
	int prev_length = 2 * previous_M + 1;
	int inc = 1;
	double *tmp = new double[prev_length + 1];
	double *tmp2 = new double[2 * DynamicM];

	double* buffer = new double[DynamicM * 2 + 2];
	memset(buffer, 0, sizeof(double) * size_t(DynamicM * 2 + 2));
	for (i=0;i<prev_length;i++)
		tmp[i] = step_size * ddot_(&prev_length, t, &inc, inner_product_matrix[i], &inc);
	tmp[prev_length] = step_size * ddot_(&prev_length, t, &inc, tmp, &inc);

	for (i=0;i<DynamicM;i++)
	{
		buffer[2 * i] = ddot_(&length, g + start, &inc, u[i], &inc);
		buffer[2 * i + 1] = ddot_(&length, g + start, &inc, s[i], &inc);
	}
	buffer[2 * DynamicM] = ddot_(&length, g + start, &inc, g + start, &inc);
	buffer[2 * DynamicM + 1] = ddot_(&length, s[k], &inc, s[k], &inc);
	mpi_allreduce(buffer, 2 * DynamicM + 2, MPI_DOUBLE, MPI_SUM);
	for (i=0;i<prev_length - 1; i++)
		tmp2[i] = inner_product_matrix[prev_length-1][i] - buffer[i] ;

	tmp2[2 * k] = tmp[prev_length - 1] - buffer[2 * k];
//	u[k] * s[k] = u[k] * g - u[k] * old_g

	tmp2[2 * k + 1] = buffer[2 * DynamicM + 1];
//	s[k] * s[k]

	for (i=0;i< prev_length - 1;i++)
	{
		inner_product_matrix[2 * k][i] = tmp[i];
		inner_product_matrix[i][2 * k] = tmp[i];
	}

	for (i=0;i< 2 * DynamicM;i++)
	{
		inner_product_matrix[2 * k + 1][i] = tmp2[i];
		inner_product_matrix[i][2 * k + 1] = tmp2[i];
	}

	inner_product_matrix[2 * k][2 * k] = tmp[prev_length];

	for (i=0; i< 2 * DynamicM + 1; i++)
	{
		inner_product_matrix[2 * DynamicM][i] = buffer[i];
		inner_product_matrix[i][2 * DynamicM] = buffer[i];
	}

	delete[] buffer;
	delete[] tmp;
	delete[] tmp2;
}

class l2r_lr_fun: public l2r_erm_fun
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	void Hessian_coeff(double *C_times_D);

private:
	double *D;
	double C_times_loss(int i, double wx_i);
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C):
	l2r_erm_fun(prob, C)
{
	int l=prob->l;
	D = new double[l];
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] D;
}

double l2r_lr_fun::C_times_loss(int i, double wx_i)
{
	double ywx_i = wx_i * prob->y[i];
	if (ywx_i >= 0)
		return C[i]*log(1 + exp(-ywx_i));
	else
		return C[i]*(-ywx_i + log(1 + exp(ywx_i)));
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y = prob->y;
	int l = prob->l;
	int w_size = get_nr_variable();

	for(i=0;i<l;i++)
	{
		tmp[i] = 1 / (1 + exp(-y[i] * wx[i]));
		D[i] = tmp[i] * (1 - tmp[i]);
		tmp[i] = C[i] * (tmp[i] - 1) * y[i];
	}
	XTv(tmp, g);
	for (i=start;i<start + length;i++)
		g[i] = w[i] + g[i];
	mpi_allreduce(g, w_size, MPI_DOUBLE, MPI_SUM);
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i] = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi=x[i];
		double xTs = sparse_operator::dot(s, xi);

		xTs = C[i]*D[i]*xTs;

		sparse_operator::axpy(xTs, xi, Hs);
	}
	for(i=start;i<start + length;i++)
		Hs[i] = s[i] + Hs[i];
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
}

void l2r_lr_fun::Hessian_coeff(double *C_times_D)
{
	int l=prob->l;
	for (int i=0;i<l;i++)
		C_times_D[i] = C[i] * D[i];
}

class l2r_l2_svc_fun: public l2r_erm_fun
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	void Hessian_coeff(double *C_times_D);

protected:
	void subXTv(double *v, double *XTv);

	int *I;
	int sizeI;
private:
	double C_times_loss(int i, double wx_i);
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C):
	l2r_erm_fun(prob, C)
{
	I = new int[prob->l];
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] I;
}

double l2r_l2_svc_fun::C_times_loss(int i, double wx_i)
{
		double d = 1 - prob->y[i] * wx_i;
		if (d > 0)
			return C[i] * d * d;
		else
			return 0;
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
	{
		tmp[i] = wx[i] * y[i];
		if (tmp[i] < 1)
		{
			tmp[sizeI] = 2 * C[i]*y[i]*(tmp[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	}
	subXTv(tmp, g);
	for(i=start;i<start + length;i++)
		g[i] = w[i] + g[i];
	mpi_allreduce(g, w_size, MPI_DOUBLE, MPI_SUM);
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		double xTs = sparse_operator::dot(s, xi);

		xTs = 2 * C[I[i]]*xTs;

		sparse_operator::axpy(xTs, xi, Hs);
	}
	for(i=start;i<start + length;i++)
		Hs[i] = s[i] + Hs[i];
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}

void l2r_l2_svc_fun::Hessian_coeff(double *C_times_D)
{
	int l=prob->l;
	for (int i=0;i<l;i++)
		C_times_D[i] = 0;
	for (int i=0;i<sizeI;i++)
		C_times_D[I[i]] = C[I[i]];
}

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
	l2r_l2_svr_fun(const problem *prob, double *C, double p);

	void grad(double *w, double *g);

private:
	double C_times_loss(int i, double wx_i);
	double p;
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *prob, double *C, double p):
	l2r_l2_svc_fun(prob, C)
{
	this->p = p;
}

double l2r_l2_svr_fun::C_times_loss(int i, double wx_i)
{
		double d = wx_i - prob->y[i];
		if(d < -p)
			return C[i]*(d+p)*(d+p);
		else if(d > p)
			return C[i]*(d-p)*(d-p);
		return 0;
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = wx[i] - y[i];

		// generate index set I
		if(d < -p)
		{
			tmp[sizeI] = 2 * C[i]*(d+p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > p)
		{
			tmp[sizeI] = 2 * C[i]*(d-p);
			I[sizeI] = i;
			sizeI++;
		}

	}
	subXTv(tmp, g);

	for(i=start;i<start + length;i++)
		g[i] = w[i] + g[i];
	mpi_allreduce(g, w_size, MPI_DOUBLE, MPI_SUM);
}

// A coordinate descent algorithm for
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
//
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i,
//  C^m_i = 0 if m != y_i,
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i
//
// Given:
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) ((int) prob->y[i])
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *weighted_C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = weighted_C;
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];
	beta /= r;

	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[GETI(i)];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;

	// Initial alpha can be set here. Note that
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;
		feature_node *xi = prob->x[i];
		QD[i] = 0;
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;

			// Uncomment the for loop if initial alpha isn't zero
			// for(m=0; m<nr_class; m++)
			//	w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
			xi++;
		}
		active_size_i[i] = nr_class;
		y_index[i] = (int)prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter)
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi->value);
					xi++;
				}

				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[(int) prob->y[i]] < C[GETI(i)] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i],
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m)
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+(int)prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//		upper_bound_i = Cp if y_i = 1
//		upper_bound_i = Cn if y_i = -1
//		D_ii = 0
// In L2-SVM case:
//		upper_bound_i = INF
//		D_ii = 1/(2*Cp)	if y_i = 1
//		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 1 of Lee and Roth, 2015

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static inline double l1_loss (double xi)
{
	return xi;
}

static inline double l2_loss (double xi)
{
	return xi*xi;
}

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int max_inner_iter = 1;
	int *index = new int[l];
	double *alpha = new double[l];
	double *alpha_orig = new double[l];
	double *alpha_inc = new double[l];
	double *w_orig = new double[w_size];
	double *current_w = new double[w_size];
	double *allreduce_buffer = new double[w_size + 3];
	double old_primal, primal, obj, grad_alpha_inc;
	double lambda = 0;
	double loss, reg = 0;
	schar *y = new schar[l];
	double eta;
	double init_primal = 0;
	static double (*loss_term) (const double) = &l2_loss;
	double sum_alpha_inc;
	double w_inc_square;
	double w_dot_w_inc;
	double alpha_inc_square;
	double alpha_inc_dot_alpha;
	double max_step;

	// PG: projected gradient
	double PG;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	double Cs[3] = {Cn, 0, Cp};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		loss_term = &l1_loss;
		lambda = 1e-3;
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
			init_primal += Cp;
		}
		else
		{
			y[i] = -1;
			init_primal += Cn;
		}
	}
	mpi_allreduce(&init_primal, 1, MPI_DOUBLE, MPI_SUM);

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;

	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)] + lambda;

		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[i], xi, w);

		index[i] = i;
	}

	old_primal = INF;
	obj = 0;

// uncomment the following lines if the initial alpha is not 0
/*
	mpi_allreduce(w, w_size, MPI_DOUBLE, MPI_SUM);
	memcpy(current_w, w, sizeof(double) * w_size);
	for (i=0;i<w_size;i++)
		reg += w[i] * w[i];
	reg *= 0.5;
	for (i=0;i<l;i++)
	{
		obj += alpha[i] * (alpha[i] * diag[GETI(i)] - 2);
		feature_node const *xi = prob->x[i];
		loss = 1 - y[i] * sparse_operator::dot(w, xi);

		if (loss > 0)
			old_primal += loss_term(loss) * Cs[GETI(i)];
	}
	mpi_allreduce(&old_primal, 1, MPI_DOUBLE, MPI_SUM);
	mpi_allreduce(&obj, 1, MPI_DOUBLE, MPI_SUM);
	old_primal += reg;
	obj = obj / 2 + reg;
*/

	while (iter < max_iter)
	{
		memcpy(w_orig, w, sizeof(double) * w_size);
		memcpy(alpha_orig, alpha, sizeof(double) * l);
		memset(alpha_inc, 0, sizeof(double) * l);
		max_step = INF;
		sum_alpha_inc = 0;
		w_inc_square = 0;
		w_dot_w_inc = 0;
		alpha_inc_square = 0;
		alpha_inc_dot_alpha = 0;

		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}

		for (int inner_iter = 0;inner_iter < max_inner_iter; inner_iter++)
		{
			for (s=0; s<l; s++)
			{
				i = index[s];
				const schar yi = y[i];
				feature_node const *xi = prob->x[i];

				G = yi*sparse_operator::dot(w, xi)-1;

				C = upper_bound[GETI(i)];
				G += alpha[i]*diag[GETI(i)];

				PG = 0;
				if (alpha[i] == 0)
				{
					if (G < 0)
						PG = G;
				}
				else if (alpha[i] == C)
				{
					if (G > 0)
						PG = G;
				}
				else
					PG = G;

				if(fabs(PG) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
					d = yi*(alpha[i] - alpha_old);
					xi = prob->x[i];
					sparse_operator::axpy(d, xi, w);
				}
			}
		}

		for (i=0;i<l;i++)
		{
			alpha_inc[i] = alpha[i] - alpha_orig[i];
			sum_alpha_inc += alpha_inc[i];
			alpha_inc_square += alpha_inc[i] * alpha_inc[i] * diag[GETI(i)];
			alpha_inc_dot_alpha += alpha_inc[i] * alpha_orig[i] * diag[GETI(i)];
			if (alpha_inc[i] > 0)
				max_step = min(max_step, (upper_bound[GETI(i)] - alpha_orig[i]) / alpha_inc[i]);
			else if (alpha_inc[i] < 0)
				max_step = min(max_step, - alpha_orig[i] / alpha_inc[i]);
		}

		for (i=0;i<w_size;i++)
			allreduce_buffer[i] = w[i] - w_orig[i];

		allreduce_buffer[w_size] = sum_alpha_inc;
		allreduce_buffer[w_size + 1] = alpha_inc_square;
		allreduce_buffer[w_size + 2] = alpha_inc_dot_alpha;
		mpi_allreduce(allreduce_buffer, w_size + 3, MPI_DOUBLE, MPI_SUM);

		for (i=0;i<w_size;i++)
		{
			w_inc_square += allreduce_buffer[i] * allreduce_buffer[i];
			w_dot_w_inc += allreduce_buffer[i] * w_orig[i];
		}

		sum_alpha_inc = allreduce_buffer[w_size];
		alpha_inc_square = allreduce_buffer[w_size + 1];
		alpha_inc_dot_alpha = allreduce_buffer[w_size + 2];
		mpi_allreduce(&max_step, 1, MPI_DOUBLE, MPI_MIN);

		grad_alpha_inc = w_dot_w_inc + alpha_inc_dot_alpha - sum_alpha_inc;
		if (grad_alpha_inc >= 0)
		{
			memcpy(w, current_w, sizeof(double) * w_size);
			break;
		}

		double aQa = alpha_inc_square + w_inc_square;
		eta = min(max_step, -grad_alpha_inc /aQa);
		if (eta <= 0)
		{
			memcpy(w, w_orig, sizeof(double) * w_size);
			memcpy(alpha, alpha_orig, sizeof(double) * l);
			info("WARNING: Negative step faced\n");
			break;
		}


		for (i=0;i<w_size;i++)
			w[i] = w_orig[i] + eta * allreduce_buffer[i];
		for (i=0;i<l;i++)
			alpha[i] = alpha_orig[i] + eta * alpha_inc[i];

		obj += eta * (0.5 * eta * aQa + grad_alpha_inc);

		reg += eta * (w_dot_w_inc + 0.5 * eta * w_inc_square);
		primal = 0;

		for (i=0;i<l;i++)
		{
			feature_node const *xi = prob->x[i];
			loss = 1 - y[i] * sparse_operator::dot(w, xi);

			if (loss > 0)
				primal += Cs[GETI(i)] * loss_term(loss);
		}
		mpi_allreduce(&primal, 1, MPI_DOUBLE, MPI_SUM);

		primal += reg;

		if (primal < old_primal)
		{
			old_primal = primal;
			memcpy(current_w, w, sizeof(double) * w_size);
		}

		iter++;
		double gap = (primal+obj) / init_primal;
		info("iter %3d primal %5.3e dual %5.3e duality gap %5.3e\n", iter, primal, obj, gap);
		if (gap < eps)
		{
			memcpy(w, current_w, sizeof(double) * w_size);
			break;
		}
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n\n");

	// calculate objective value

	int nSV = 0;
	for(i=0; i<l; i++)
		if(alpha[i] > 0)
			++nSV;
	mpi_allreduce(&nSV, 1, MPI_INT, MPI_SUM);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] index;
	delete [] alpha;
	delete [] alpha_inc;
	delete [] alpha_orig;
	delete [] w_orig;
	delete [] current_w;
	delete [] allreduce_buffer;
	delete [] y;
}


// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//		upper_bound_i = C
//		lambda_i = 0
// In L2-SVM case:
//		upper_bound_i = INF
//		lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
	const problem *prob, double *w, const parameter *param,
	int solver_type)
{
	int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(beta[i], xi, w);

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node * const xi = prob->x[i];
			G += sparse_operator::dot(w, xi);

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
				sparse_operator::axpy(d, xi, w);
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}


// A coordinate descent algorithm for
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2;
	double innereps_min = min(1e-8, eps);
	double upper_bound[3] = {Cn, 0, Cp};

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		xTx[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}
		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const schar yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			feature_node * const xi = prob->x[i];
			ywTx = yi*sparse_operator::dot(w, xi);
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, xi, w);
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	for(i=0; i<w_size; i++)
		v += w[i] * w[i];
	v *= 0.5;
	for(i=0; i<l; i++)
		v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
			- upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	info("Objective value = %lf\n", v);

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x->value *= y[ind]; // x->value stores yi*xij
			double val = x->value;
			b[ind] -= w[j]*val;
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*w[j])
				d = -Gp/H;
			else if(Gn > H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					sparse_operator::axpy(d_diff, x, b);
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					sparse_operator::axpy(-w[i], x, b);
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
	int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				sparse_operator::axpy(z, x, xTd);
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				sparse_operator::axpy(w[i], x, exp_wTx);
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		info("iter %3d  #CD cycles %d\n", newton_iter, iter);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

// transpose matrix X from row format to column format
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int i;

	std::set<int> label_set;
	for(i=0;i<prob->l;i++)
		label_set.insert((int)prob->y[i]);
	
	int label_size = (int)label_set.size();
	int num_machines = mpi_get_size();
	int max_label_size;
	MPI_Allreduce(&label_size, &max_label_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	std::vector<int> global_label_sets((max_label_size+1)*num_machines);
	std::vector<int> label_buff(max_label_size+1);

	label_buff[0] = label_size;
	i = 1;
	for(std::set<int>::iterator this_label=label_set.begin();
			this_label!=label_set.end(); this_label++)
	{
		label_buff[i] = (*this_label);
		i++;
	}
	
	MPI_Allgather(label_buff.data(), max_label_size+1, MPI_INT, global_label_sets.data(), max_label_size+1, MPI_INT, MPI_COMM_WORLD);

	for(i=0; i<num_machines; i++)
	{
		int offset = i*(max_label_size+1);
		int size = global_label_sets[offset];
		for(int j=1; j<=size; j++)
			label_set.insert(global_label_sets[offset+j]);
	}

	int nr_class = (int)label_set.size();

	std::map<int, int> label_map;
	int *label = Malloc(int, nr_class);
	i = 0;
	for(std::set<int>::iterator this_label=label_set.begin();
			this_label!=label_set.end(); this_label++)
	{
		label[i] = (*this_label);
		i++;
	}
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
		swap(label[0], label[1]);
	for(i=0;i<nr_class;i++)
		label_map[label[i]] = i;


	// The following codes are similar to the original LIBLINEAR
	int *start = Malloc(int, nr_class);
	int *count = Malloc(int, nr_class);
	for(i=0;i<nr_class;i++)
		count[i] = 0;
	for(i=0;i<prob->l;i++)
		count[label_map[(int)prob->y[i]]]++;

	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[label_map[(int)prob->y[i]]]] = i;
		++start[label_map[(int)prob->y[i]]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	//inner and outer tolerances for TRON
	double eps = param->eps;
	double eps_cg = 0.1;
	if(param->init_sol != NULL)
		eps_cg = 0.5;

	int l = prob->l;
	int pos = 0;
	int neg = 0;
	for(int i=0; i<l; i++)
		if(prob->y[i] > 0)
			pos++;

	mpi_allreduce(&pos, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&l, 1, MPI_INT, MPI_SUM);
	neg = l - pos;

	double primal_solver_tol = (eps*max(min(pos,neg), 1))/l;

	l2r_erm_fun *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2R_LR:
		{
			double *C = new double[l];
			for(int i = 0; i < l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_LR_COMM:
		{
			double *C = new double[l];
			for(int i = 0; i < l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			L_COMMDIR_BFGS commdir_obj(fun_obj, l, primal_solver_tol, param->m);
			commdir_obj.l_commdir_bfgs(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[l];
			for(int i = 0; i < l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC_COMM:
		{
			double *C = new double[l];
			for(int i = 0; i < l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			L_COMMDIR_BFGS commdir_obj(fun_obj, l, primal_solver_tol, param->m);
			commdir_obj.l_commdir_bfgs(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_l2_svc(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		case L2R_L2LOSS_SVR:
		{
			double *C = new double[l];
			for(int i = 0; i < l; i++)
				C[i] = param->C;

			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
			TRON tron_obj(fun_obj, param->eps, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L2LOSS_SVR_DUAL);
			break;
		default:
		{
			if(mpi_get_rank() == 0)
				fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
		}
	}
}

// Calculate the initial C for parameter selection
static double calc_start_C(const problem *prob, const parameter *param)
{
	int i;
	double xTx,max_xTx;
	max_xTx = 0;
	for(i=0; i<prob->l; i++)
	{
		xTx = 0;
		feature_node *xi=prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			xTx += val*val;
			xi++;
		}
		if(xTx > max_xTx)
			max_xTx = xTx;
	}

	double min_C = 1.0;
	if(param->solver_type == L2R_LR || param->solver_type == L2R_LR_COMM)
		min_C = 1.0 / (prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVC || param->solver_type == L2R_L2LOSS_SVC_COMM)
		min_C = 1.0 / (2 * prob->l * max_xTx);

	return pow( 2, floor(log(min_C) / log(2.0)) );
}


//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(check_regression_model(model_))
	{
		model_->w = Malloc(double, w_size);
		for(i=0; i<w_size; i++)
			model_->w[i] = 0;
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, model_->w, 0, 0);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
			{
				fprintf(stderr,"[rank %d] WARNING: class label %d specified in weight is not found\n",
					mpi_get_rank(), param->weight_label[i]);
			}
			else
				weighted_C[j] *= param->weight[i];
		}

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		// multi-class svm by Crammer and Singer
		if(param->solver_type == MCSVM_CS)
		{
			model_->w=Malloc(double, n*nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;
			Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
			Solver.Solve(model_->w);
		}
		else
		{
			if(nr_class == 2)
			{
				model_->w=Malloc(double, w_size);

				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				if(param->init_sol != NULL)
					for(i=0;i<w_size;i++)
						model_->w[i] = param->init_sol[i];
				else
					for(i=0;i<w_size;i++)
						model_->w[i] = 0;

				train_one(&sub_prob, param, model_->w, weighted_C[0], weighted_C[1]);
			}
			else
			{
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					if(param->init_sol != NULL)
						for(j=0;j<w_size;j++)
							w[j] = param->init_sol[j*nr_class+i];
					else
						for(j=0;j<w_size;j++)
							w[j] = 0;

					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
				}
				free(w);
			}

		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	}
	return model_;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"[rank %d] WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n",
		        mpi_get_rank());
	}
	mpi_allreduce(&nr_fold, 1, MPI_INT, MPI_MIN);
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

void find_parameter_C(const problem *prob, const parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate)
{
	// variables for CV
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int, l);
	double *target = Malloc(double, prob->l);
	struct problem *subprob = Malloc(problem,nr_fold);

	// variables for warm start
	double ratio = 2;
	double **prev_w = Malloc(double*, nr_fold);
	for(i = 0; i < nr_fold; i++)
		prev_w[i] = NULL;
	int num_unchanged_w = 0;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;

	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;

		subprob[i].bias = prob->bias;
		subprob[i].n = prob->n;
		subprob[i].l = l-(end-begin);
		subprob[i].x = Malloc(struct feature_node*,subprob[i].l);
		subprob[i].y = Malloc(double,subprob[i].l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}

	}

	*best_rate = 0;
	if(start_C <= 0)
		start_C = calc_start_C(prob,param);
	param1.C = start_C;

	while(param1.C <= max_C)
	{
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		for(i=0; i<nr_fold; i++)
		{
			int j;
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = prev_w[i];
			struct model *submodel = train(&subprob[i],&param1);

			int total_w_size;
			if(submodel->nr_class == 2)
				total_w_size = subprob[i].n;
			else
				total_w_size = subprob[i].n * submodel->nr_class;

			if(prev_w[i] == NULL)
			{
				prev_w[i] = Malloc(double, total_w_size);
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}
			else if(num_unchanged_w >= 0)
			{
				double norm_w_diff = 0;
				for(j=0; j<total_w_size; j++)
				{
					norm_w_diff += (submodel->w[j] - prev_w[i][j])*(submodel->w[j] - prev_w[i][j]);
					prev_w[i][j] = submodel->w[j];
				}
				norm_w_diff = sqrt(norm_w_diff);

				if(norm_w_diff > 1e-15)
					num_unchanged_w = -1;
			}
			else
			{
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}

			for(j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		set_print_string_function(default_print_string);

		int total_correct = 0;
		for(i=0; i<prob->l; i++)
			if(target[i] == prob->y[i])
				++total_correct;
		double current_rate = (double)total_correct/prob->l;
		if(current_rate > *best_rate)
		{
			*best_C = param1.C;
			*best_rate = current_rate;
		}

		info("log2c=%7.2f\trate=%g\n",log(param1.C)/log(2.0),100.0*current_rate);
		num_unchanged_w++;
		if(num_unchanged_w == 3)
			break;
		param1.C = param1.C*ratio;
	}

	if(param1.C > max_C && max_C > start_C)
		info("warning: maximum C reached.\n");
	free(fold_start);
	free(perm);
	free(target);
	for(i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
		free(prev_w[i]);
	}
	free(prev_w);
	free(subprob);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(check_regression_model(model_))
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"L2R_LR_COMM", "L2R_L2LOSS_SVC_COMM", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	setlocale(LC_ALL, old_locale);\
	free(model_->label);\
	free(model_);\
	free(old_locale);\
	return NULL;\
}
struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;
	// parameters for training only won't be assigned, but arrays are assigned as NULL for safety
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;	
	param.init_sol = NULL;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"[rank %d] unknown solver type.\n", mpi_get_rank());
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"[rank %d] unknown text in model file: [%s]\n",mpi_get_rank(),cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
	int nr_class = model_->nr_class;
	int solver_type = model_->param.solver_type;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(check_regression_model(model_))
		return w[idx];
	else
	{
		if(label_idx < 0 || label_idx >= nr_class)
			return 0;
		if(nr_class == 2 && solver_type != MCSVM_CS)
		{
			if(label_idx == 0)
				return w[idx];
			else
				return -w[idx];
		}
		else
			return w[idx*nr_class+label_idx];
	}
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx)
{
	if(feat_idx > model_->nr_feature)
		return 0;
	return get_w_value(model_, feat_idx-1, label_idx);
}

double get_decfun_bias(const struct model *model_, int label_idx)
{
	int bias_idx = model_->nr_feature;
	double bias = model_->bias;
	if(bias <= 0)
		return 0;
	else
		return bias*get_w_value(model_, bias_idx, label_idx);
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
	if(param->init_sol != NULL)
		free(param->init_sol);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_LR
	  && param->solver_type != L2R_L2LOSS_SVC
	  && param->solver_type != L2R_L2LOSS_SVC_DUAL
	  && param->solver_type != L2R_L1LOSS_SVC_DUAL
	  && param->solver_type != L2R_LR_COMM
	  && param->solver_type != L2R_L2LOSS_SVC_COMM)
	 
		return "unknown solver type";

	if(param->init_sol != NULL 
		&& param->solver_type != L2R_LR && param->solver_type != L2R_L2LOSS_SVC && param->solver_type != L2R_LR_COMM && param->solver_type != L2R_L2LOSS_SVC_COMM)
		return "Initial-solution specification supported only for solvers -s 0, 2, 8, and 9";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR ||
			model_->param.solver_type==L2R_LR_COMM);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_L2LOSS_SVR ||
			model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
			model_->param.solver_type==L2R_L2LOSS_SVR_DUAL);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}
