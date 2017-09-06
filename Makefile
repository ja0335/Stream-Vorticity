# g++ -o simulation-app -Wall -Wextra -Werror -c -lsfml-graphics -lsfml-window -lsfml-system -cuda -cudart -cublas -cublas_device  

CC=nvcc
SRC := Stream-Vorticity
OBJ := obj
LIBS = -lsfml-graphics -lsfml-window -lsfml-system -lcuda -lcudart -lcublas -lcublas_device  

SOURCES_CU := $(wildcard $(SRC)/*.cu)
OBJECTS_CU := $(patsubst $(SRC)/%.cu, $(OBJ)/%.o, $(SOURCES_CU))

SOURCES_CPP := $(wildcard $(SRC)/*.cpp)
OBJECTS_CPP := $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(SOURCES_CPP))


Stream-Vorticity.app: $(OBJECTS_CU) $(OBJECTS_CPP)
	$(CC) -std=c++11 --gpu-architecture=sm_30 $^ -o $@ $(LIBS)

$(OBJ)/%.o: $(SRC)/%.cu
	$(CC) -I$(SRC) -c -std=c++11 --gpu-architecture=sm_30 $< -o $@ $(LIBS)

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CC) -I$(SRC) -c -std=c++11 --gpu-architecture=sm_30 $< -o $@ $(LIBS)
