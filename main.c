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

#define EXAMPLES         4   //liczba przykładów podanych sieci
#define NUM_INPUT        2   //liczba wejść
#define NUM_HIDDEN       2   //liczba ukrytych neuronów
#define NUM_OUTPUT       1   //liczba wyjść z sieci
#define NUM_BIAS         3   //liczba "biasów"
#define NUM_LINK         9   //liczba połączeń w sieci
#define ALPHA          0.1   //współczynnik nauki sieci
#define BIAS          -1.0   //wartość bias
#define ITERATION   300000   //ilość iteracji podczas nauki

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

void connectLayers(struct Neuron neuronPickFirst[], struct Neuron neuronPickSecond[], \
                   struct Link *link[], int lCount, int bCount, \
                   int x, int y){

  int i = lCount;


  if(bCount == 0){
      for(int j = 0; j < x; j++){
        for(int k = 0; k < y; k++){
          link[i]->wage = (((double)(rand()%100)+0.1) / ((double)(rand()%100)+0.1));
          printf("%d : %lf\n", i, link[i]->wage);
          link[i]->from = &(neuronPickFirst[j]);
          link[i]->to   = &(neuronPickSecond[k]);
          i++;
        }
      }
  }
  else printf("ERROR\n");

  return;
}

void connectBias(struct Neuron neuronPickFirst[], struct Neuron neuronPickSecond[], \
                 struct Link *link[], int lCount, int bCount, \
                 int x, int y){

  int i = lCount;
  int k = 0;
  int j = bCount;

  while(j < x && k < y){
    link[i]->wage = (((double)(rand()%100)+0.1) / ((double)(rand()%100)+0.1));
    printf("%d : %lf\n", i, link[i]->wage);
    link[i]->from = &(neuronPickFirst[j]);
    link[i]->to   = &(neuronPickSecond[k]);
    i++;
    j++;
    k++;
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

  //liczniki tworzonych połączeń
  int tmpLinkCount = 0;
  int tmpBiasCount = 0;

  //ziarno do losowania
  srand(time(NULL));

  connectLayers(neuronInput, neuronHidden, link, tmpLinkCount, tmpBiasCount, NUM_INPUT, NUM_HIDDEN);
  tmpLinkCount += (NUM_INPUT * NUM_HIDDEN);

  connectLayers(neuronHidden, neuronOutput, link, tmpLinkCount, tmpBiasCount, NUM_HIDDEN, NUM_OUTPUT);
  tmpLinkCount += (NUM_HIDDEN * NUM_OUTPUT);

  connectBias(neuronBias, neuronHidden, link, tmpLinkCount, tmpBiasCount, NUM_BIAS, NUM_HIDDEN);
  tmpLinkCount += NUM_HIDDEN;
  tmpBiasCount += NUM_HIDDEN;

  connectBias(neuronBias, neuronOutput, link, tmpLinkCount, tmpBiasCount, NUM_BIAS, NUM_OUTPUT);
  tmpLinkCount += NUM_OUTPUT;
  tmpBiasCount += NUM_OUTPUT;

///////////Proces Nauki///////////

  double networkError = 0.0;
  int    randomizeTab[EXAMPLES]; //tablica przechowująca wynik losowego wybrania wektoru uczącego
  int    flag = 0;
  int    num  = 0;

  srand(time(NULL));

  for(int x=0; x<(ITERATION * EXAMPLES); x++){    //ilość iteracji razy ilość przykładów uczących

    //losowanie przykładu do sieci
      if((x%EXAMPLES)==0){
        for(int i=0; i<EXAMPLES; i++) randomizeTab[i] = 0;
      }

      do{
        num = rand()%EXAMPLES;

        if(x%EXAMPLES==0){                      //jeżeli epoka jest wielokrotnością ilości wektorów uczących to wstawia
          flag = 1;
          randomizeTab[x%EXAMPLES] = num;
        }
        else{                                   //inaczej przegląda tablicę użytych i szuka niewykorzystanego
          for(int i=0; i<(x%EXAMPLES);i++){
            if(randomizeTab[i]==num){
              flag = 0;
              break;                            //jeżeli trafił na użyty wychodzi z pętli i losuje od nowa
            }
            else{
              flag = 1;
            }
          }
          if(flag==1) randomizeTab[x%EXAMPLES] = num;
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
    if(networkError<0.0003 && networkError>(-0.0003)){
      printf("Odbyło się %d iteracji.\n", x);
      break;
    }

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

  printf("\n\033[01;36m---Proces nauki zakończony---\033[01;37m\n\n");

  while(1){

    printf("Chcesz zakończyć działanie? (y | n) : ");
    scanf("%c", &esc);
    if(esc=='y'){
      printf("\n\033[01;36m---Zamknięcie programu---\x1b[0m\n");
      break;
    }
    printf("\nPodaj pierwszy składnik: ");
    scanf("%lf", &x1);
    neuronInput[0].wage = x1;
    printf("\nPodaj drugi składnik:    ");
    scanf("%lf", &x2);
    neuronInput[1].wage = x2;

    //obliczenia dla neuronów
    calculateNeuron(link, neuronHidden, 0);
    calculateNeuron(link, neuronHidden, 1);
    calculateNeuron(link, neuronOutput, 0);

    printf("\n|||||| Wynik sieci:    \033[01;35m%lf\033[01;37m\n\n", neuronOutput[0].wage);

    getchar();

  }

  return 0;
}
