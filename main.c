/*
  implementacja algorytmu wstecznej propagacji
        oraz zastosowanie w rozwiązaniu
        problemu alternatywy rozłącznej
    --------------------------------------
    |||| Jan Iwaszkiewicz @illumitata ||||
    --------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EXAMPLES       4   //liczba przykładów podanych sieci
#define NUM_INPUT      2   //liczba wejść
#define NUM_HIDDEN     2   //liczba ukrytych neuronów
#define NUM_OUTPUT     1   //liczba wyjść z sieci
#define NUM_BIAS       3   //liczba "biasów"
#define NUM_LINK       9   //liczba połączeń w sieci
#define ALPHA        0.1   //współczynnik nauki sieci
#define BIAS        -1.0   //wartość bias
#define ITERATION 250000   //ilość iteracji podczas nauki

struct Vector{          //struktura wektora uczącego
  double first;         //pierwszy składnik
  double second;        //drugi składnik
  double result;        //wynik operacji XOR
};

struct Neuron{          //struktura pojedyńczego neuronu sieci
  double wage;          //waga na wyjściu z neuronu
  double smallDelta;    //mała delta do obliczeń błędu
};

struct Link{            //struktura połączenia między dwoma neuronami
  double wage;          //waga połączenia
  double bigDelta;      //duża detla do obliczenia poprawki
  struct Neuron *from;  //gdzie zaczyna się połączenie
  struct Neuron *to;    //gdzie kończy się połączenie
};

//funkcja sigmoidalna unipolarna
double funcSig(double s){

  double func = 0;

  func = 1 / (1 + exp(-1 * s));

  return func;
}

//pochodna funckji sigmoidalnej unipolarnej
double funcSigDerivative(double s){

  double funcDer = 0;

  funcDer = funcSig(s) * (1 - (funcSig(s)));

  return funcDer;
}

//obliczanie danego neuronu
void calculateNeuron(struct Link *link[], struct Neuron neuronPicked[], int k){

  double tmpWage = 0.0;

  for(int i=0; i<NUM_LINK; i++){
    if(link[i]->to == &(neuronPicked[k])) tmpWage += (link[i]->from->wage) * (link[i]->wage);
  }

  neuronPicked[k].wage = funcSig(tmpWage);

  return;
}

void calculateOutputError(double error, struct Neuron neuronPicked[], int k){

  neuronPicked[k].smallDelta = error * funcSigDerivative(neuronPicked[k].wage);

  return;
}

void calculateHiddenError(struct Link *link[], struct Neuron neuronPicked[], int k){

  for(int i=0; i<NUM_LINK; i++){
    if(link[i]->from == &(neuronPicked[k])){
      neuronPicked[k].smallDelta = link[i]->wage * link[i]->to->smallDelta \
                                   * funcSigDerivative(neuronPicked[k].wage);
    }
  }

  return;
}

void backPropagation(struct Link *link[]){

  for(int i=0; i<NUM_LINK; i++){
    link[i]->bigDelta = ALPHA * (link[i]->from->wage) * (link[i]->to->smallDelta);
    link[i]->wage += link[i]->bigDelta;
  }

  return;
}

int main(){

  //inicjowanie przykładów czyli wektorów do nauki
  struct Vector vector[EXAMPLES];

  //1 1 0
  vector[0].first  = 1.0;
  vector[0].second = 1.0;
  vector[0].result = 0.0;

  //1 0 1
  vector[1].first  = 1.0;
  vector[1].second = 0.0;
  vector[1].result = 1.0;

  //0 0 0
  vector[2].first  = 0.0;
  vector[2].second = 0.0;
  vector[2].result = 0.0;

  //0 1 1
  vector[3].first  = 0.0;
  vector[3].second = 1.0;
  vector[3].result = 1.0;

  //stworzenie neuronów sieci
  struct Neuron neuronInput[NUM_INPUT];
  struct Neuron neuronHidden[NUM_HIDDEN];
  struct Neuron neuronOutput[NUM_OUTPUT];

  //stworzenie neuronów biasowych i zainicjowanie wagi -1.0
  struct Neuron neuronBias[NUM_BIAS];
  for(int i=0; i<NUM_BIAS; i++) neuronBias[i].wage = BIAS;

  //stworzenie połączeń pomiędzy neuronami
  struct Link *link[NUM_LINK];
  for(int i=0; i<NUM_LINK; i++) link[i] = calloc(1, sizeof(struct Link));

  //pierwszy input do ukrytej
  link[0]->wage = 0.5;
  link[0]->from = &(neuronInput[0]);
  link[0]->to   = &(neuronHidden[0]);

  link[1]->wage = 0.9;
  link[1]->from = &(neuronInput[0]);
  link[1]->to   = &(neuronHidden[1]);

  //drugi input do ukrytej
  link[2]->wage = 0.4;
  link[2]->from = &(neuronInput[1]);
  link[2]->to   = &(neuronHidden[0]);

  link[3]->wage = 1.0;
  link[3]->from = &(neuronInput[1]);
  link[3]->to   = &(neuronHidden[1]);

  //bias do ukrytej
  link[6]->wage = 0.8;
  link[6]->from = &(neuronBias[0]);
  link[6]->to   = &(neuronHidden[0]);

  link[7]->wage = -0.1;
  link[7]->from = &(neuronBias[1]);
  link[7]->to   = &(neuronHidden[1]);

  //ukryta do wyjściowej
  link[4]->wage = -1.2;
  link[4]->from = &(neuronHidden[0]);
  link[4]->to   = &(neuronOutput[0]);

  link[5]->wage = 1.1;
  link[5]->from = &(neuronHidden[1]);
  link[5]->to   = &(neuronOutput[0]);

  //bias do wyjściowej
  link[8]->wage = 0.3;
  link[8]->from = &(neuronBias[2]);
  link[8]->to   = &(neuronOutput[0]);

///////////Proces Nauki///////////

  double networkError = 0.0;
  int    randomizeTab[4] = {0,0,0,0}; //tablica przechowująca wynik losowego wybrania wektoru uczącego
  int    flag = 0;
  int    num  = 0;

  srand(time(NULL));

  for(int x=0; x<(ITERATION * EXAMPLES); x++){    //ilość iteracji razy ilość przykładów uczących

    //losowanie przykładu do sieci
      if((x%4)==0){
        for(int i=0; i<4; i++) randomizeTab[i] = 0;
      }

      do{
        num = rand()%4;

        if(x%4==0){                             //jeżeli epoka jest wielokrotnością ilości wektorów uczących to wstawia
          flag = 1;
          randomizeTab[x%4] = num;
        }
        else{                                   //inaczej przegląda tablicę użytych i szuka niewykorzystanego
          for(int i=0; i<(x%4);i++){
            if(randomizeTab[i]==num){
              flag = 0;
              break;                            //jeżeli trafił na użyty wychodzi z pętli i losuje od nowa
            }
            else{
              flag = 1;
            }
          }
          if(flag==1) randomizeTab[x%4] = num;
        }
      }while(flag!=1);

    //podanie sieci przykładu
    neuronInput[0].wage = vector[num].first;
    neuronInput[1].wage = vector[num].second;


    //obliczenia dla neuronów
    calculateNeuron(link, neuronHidden, 0);
    calculateNeuron(link, neuronHidden, 1);
    calculateNeuron(link, neuronOutput, 0);

    //obliczenie błędu wyjścia z sieci
    networkError = vector[num].result - neuronOutput[0].wage;

    /*
        //Funkcje pomagające w obserwacji działania sieci
        printf("-----ITERACJA %d num: %d------NETWORK ERROR: %lf\n", x, num, networkError);
        printf("A: %lf B: %lf C:%lf\n", neuronInput[0].wage, neuronInput[1].wage, vector[num].result);
        printf("Output 0: %lf\n", neuronOutput[0].wage);
    */

    //sprawdzić czy błąd jest mniejszy niż oczekiwany;
    if(networkError<0.0003 && networkError>0.0) break;

    //obliczenie błędu dla warstwy wyjściowej oraz ukrytej
    calculateOutputError(networkError, neuronOutput, 0);

    calculateHiddenError(link, neuronHidden, 0);
    calculateHiddenError(link, neuronHidden, 1);

    //propagowanie błędu czyli poprawa wartości połączeń
    backPropagation(link);

    //powtarzamy dla innych wektorów
  }

///////////Sieć zamrożona, można sprawdzić działanie///////////

  char esc;
  double x1, x2;

  printf("\n---Proces nauki zakończony---\n");

  while(1){

    printf("Chcesz zakończyć działanie? (y | n) : ");
    scanf("%c", &esc);
    if(esc=='y') break;

    printf("\nPodaj pierwszy składnik: ");
    scanf("%lf", &x1);
    neuronInput[0].wage = x1;
    printf("\nPodaj drugi składnik: ");
    scanf("%lf", &x2);
    neuronInput[1].wage = x2;

    //obliczenia dla neuronów
    calculateNeuron(link, neuronHidden, 0);
    calculateNeuron(link, neuronHidden, 1);
    calculateNeuron(link, neuronOutput, 0);

    printf("|||||| Wynik sieci: %lf\n\n", neuronOutput[0].wage);

    getchar();

  }

  return 0;
}
