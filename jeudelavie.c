// Jeu de la vie avec sauvegarde de quelques itérations
// compiler avec gcc -O3 -march=native (et -fopenmp si OpenMP souhaité)
//mpicc -Wall -O3 -march=native -fopenmp jeudelavie.c -o jeudelavie
//export OMP_NUM_THREADS=2
//mpirun -n 2 jeudelavie
//ssh lcpages@uds-505196.ad.unistra.fr

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

// hauteur et largeur de la matrice
#define HM 1200
#define LM 800
//1200
//800

// nombre total d'itérations
#define ITER 10001
//10001
// multiple d'itérations à sauvegarder
#define SAUV 1000
//1000
#define DIFFTEMPS(a,b) \
(((b).tv_sec - (a).tv_sec) + ((b).tv_usec - (a).tv_usec)/1000000.)

/* tableau de cellules */
typedef char Tab[HM][LM];

// initialisation du tableau de cellules
void init(Tab);

// calcule une nouveau tableau de cellules à partir de l'ancien
// - paramètres : ancien, nouveau
void calcnouv(Tab, Tab);

// -------------------- Nouvelles Fonctions ----------------------

//Envoie une notification pour rendre compte de l'état de la matrice local
//sur ses extrémités HAUT BAS
void envoyer_notif(Tab);

//Permet d'attendre l'actualisation envoyée par les autres processus
void recevoir_notif(Tab);

int calc_max_y(int rank);

// variables globales : pas de débordement de pile
Tab t1, t2;
Tab tsauvegarde[1+ITER/SAUV];
int rank, size;
int max_rows;



int main()
{
    MPI_Init(NULL,NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    max_rows = (size <=1 ) ? HM-1:calc_max_y(rank); //(HM/size)+2

  struct timeval tv_init, tv_beg, tv_end, tv_save;

  gettimeofday( &tv_init, NULL);

  init(t1);//On calcul t1 "localement" pour chaque processus

  gettimeofday( &tv_beg, NULL);


      for(int i=0 ; i<ITER ; i++)
      {

        if( i%2 == 0)
        {
          calcnouv(t1, t2);
          if(size > 1)envoyer_notif(t2);
          if(size > 1)recevoir_notif(t2);
        }
        else
        {
          calcnouv(t2, t1);
          if(size > 1)envoyer_notif(t1);
          if(size > 1)recevoir_notif(t1);
        }

        if(i%SAUV == 0)
        {
          printf("sauvegarde (%d)\n", i);
          // copie t1 dans tsauvegarde[i/SAUV]
          for(int x=0 ; x< ((size == 1)?(max_rows+1):max_rows-1); x++)
            for(int y=0 ; y<LM ; y++)
              tsauvegarde[i/SAUV][x][y] = t1[x][y];
        }
      }

      gettimeofday( &tv_end, NULL);

      FILE *f = fopen("jdlv.out", "w");
      for(int i=0 ; i<ITER ; i+=SAUV)
      {
        if(size >1) MPI_Gather(tsauvegarde[i/SAUV], (max_rows-1)*LM ,MPI_CHAR,
          tsauvegarde[i/SAUV], (max_rows-1)*LM ,MPI_CHAR, 0, MPI_COMM_WORLD);

        if(rank == 0){
          fprintf(f, "------------------ sauvegarde %d ------------------\n", i);
          for(int x=0 ; x<HM ; x++)
          {
            for(int y=0 ; y<LM ; y++)
              fprintf(f, tsauvegarde[i/SAUV][x][y]?"*":" ");
            fprintf(f, "\n");
          }
        }
      }
      fclose(f);
      gettimeofday( &tv_save, NULL);

      printf("init : %lf s,", DIFFTEMPS(tv_init, tv_beg));
      printf(" calcul : %lf s,", DIFFTEMPS(tv_beg, tv_end));
      printf(" sauvegarde : %lf s\n", DIFFTEMPS(tv_end, tv_save));


  MPI_Finalize();
  return( 0 );
}

void init(Tab t)
{

  srand(time(0));
  for(int i=0 ; i< max_rows+1; i++){
    for(int j=0 ; j<LM; j++ )
    {
      // t[i][j] = rand()%2;
      // t[i][j] = ((i+j)%3==0)?1:0;
      // t[i][j] = (i==0||j==0||i==h-1||j==l-1)?0:1;
      t[i][j] = 0;
    }
  }

    if(rank == 0){

      t[10][10] = 1;
      t[10][11] = 1;
      t[10][12] = 1;
      t[9][12] = 1;
      t[8][11] = 1;

      t[55][50] = 1;
      t[54][51] = 1;
      t[54][52] = 1;
      t[55][53] = 1;
      t[56][50] = 1;
      t[56][51] = 1;
      t[56][52] = 1;
    }

}

int nbvois(Tab t, int i, int j)
{
  int n=0;
  if( i>0 )
  {  /* i-1 */
    if( j>0 )
      if( t[i-1][j-1] )
        n++;
    if( t[i-1][j] )
        n++;
    if( j<LM-1 )
      if( t[i-1][j+1] )
        n++;
  }
  if( j>0 )
    if( t[i][j-1] )
      n++;
  if( j<LM-1 )
    if( t[i][j+1] )
      n++;
  if( i<max_rows)
  {  /* i+1 */
    if( j>0 )
      if( t[i+1][j-1] )
        n++;
    if( t[i+1][j] )
        n++;
    if( j<LM-1 )
      if( t[i+1][j+1] )
        n++;
  }
  return( n );
}

void calcnouv(Tab t, Tab n)
{
  #pragma omp parallel for
  for(int i=0 ; i< max_rows+1; i++)
    for(int j=0 ; j<LM ; j++)
    {
      int v = nbvois(t, i, j);
      if(v==3)
        n[i][j] = 1;
      else if(v==2)
        n[i][j] = t[i][j];
      else
        n[i][j] = 0;
    }

}

void envoyer_notif(Tab tab)
{
	int prev_rank = ((rank == 0) ?MPI_PROC_NULL:rank-1);
	int next_rank = ((rank == size-1) ?MPI_PROC_NULL:rank+1);

	MPI_Request request;
	MPI_Isend(&(tab[1][0]), LM, MPI_CHAR, prev_rank, 0, MPI_COMM_WORLD,&request);
	MPI_Isend(&(tab[max_rows-1][0]), LM, MPI_CHAR, next_rank, 0, MPI_COMM_WORLD, &request);
}

void recevoir_notif(Tab tab)
{
  int prev_rank = ((rank == 0) ?MPI_PROC_NULL:rank-1);
	int next_rank = ((rank == size-1) ?MPI_PROC_NULL:rank+1);
	
	//MPI_Request request;
	MPI_Recv(&tab[0][0], LM, MPI_CHAR, prev_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&tab[max_rows][0], LM, MPI_CHAR, next_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int calc_max_y(int rank) {
	
		return (int)floor((double)HM/size)+1;
	
}
