#!/bin/bash

# Script para compilar o cenário hierárquico com mobilidade Shanghai

echo "Compilando scenario-hierarchical-xangai com mobilidade Shanghai..."

# Verifica se estamos no diretório correto
if [ ! -f "scenario-hierarchical-xangai.cc" ]; then
    echo "Erro: scenario-hierarchical-xangai.cc não encontrado no diretório atual"
    exit 1
fi

# Define variáveis do ns-3
NS3_DIR="/home/elioth/ns-3-mmwave-oran"
BUILD_DIR="$NS3_DIR/build"
SCRATCH_DIR="$NS3_DIR/scratch"

# Copia os arquivos para o diretório scratch
cp scenario-hierarchical-xangai.cc "$SCRATCH_DIR/"
cp shanghai-mobility-model.h "$SCRATCH_DIR/"
cp shanghai-mobility-model.cc "$SCRATCH_DIR/"

echo "Arquivos copiados para $SCRATCH_DIR"

# Entra no diretório de build e compila
cd "$BUILD_DIR"

# Configura o waf para encontrar os novos headers
export CXXFLAGS="-I$SCRATCH_DIR $CXXFLAGS"

# Compila o cenário
./waf configure --enable-examples --enable-tests
./waf build

if [ $? -eq 0 ]; then
    echo "Compilação concluída com sucesso!"
    echo "Para executar a simulação com mobilidade Shanghai:"
    echo "./waf --run \"scenario-hierarchical-xangai --positionAllocator=2 --shanghaiScenarioId=23 --shanghaiTimeScale=1.0 --ues=10 --simTime=30\""
else
    echo "Erro na compilação"
    exit 1
fi