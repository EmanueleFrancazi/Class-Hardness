#!/usr/bin/bash

#WARNING: il programma allo stato attuale prende in input un solo parametro (numero repliche)

#WARNING: inserisco nello script (alla fine anche il comando per copiare i risultati in locale; il comando dipende dalla macchina a cui sei collegato, modificalo di conseguenza(linux20, linux19,...))

#cerca di capire come funziona per dare in input il numero di samples (perchè utilizzando $1 il problema credo stia nel fatto che non è un int (non funziona))

#INPUT PARAMETERS: the following are the set of parameters used to fix the model for the simulation

FolderName='SimulationResult'






mkdir $FolderName 


for ((i=$1; i<=$2; i++))
do
#metto un if statment in modo da non parallelizzare l'ultima replica; in questo modo sono sicuro che il trasferimento dei file da remoto a locale avvenga in maniera sequenziale all'uscita dal loop
	#if [$i -lt $1]; then
	
    	python3 LinearNet.py $i $FolderName
    	
    	#with the & option (as below) scripts run in parallel
		#python3 TestNN.py $i $FolderName $Dataset  &


		
	#else
	#	python3 TestNN.py $i $FolderName
	#fi 
	#fi conclude l'if statement
done






#assegno 2 variabili per il path remoto e locale
#Rpwd=$(pwd)
#Lpwd=/home/emanuele/Desktop/prova

#TotRpwd=${Rpwd}/SimulationResult/
#echo "$TotRpwd"

#scp -r francaem@siam-linux20:/blabla blablalocale

#I recommend rsync (instead of scp) because you can resume transfers if the connection breaks, and it intelligently transfers only the differences between files

#questa linea non va bene perchè la macchina remota non conosce il tuo ip devi creare un altro script tenendolo in localen che copi i file dal remoto al locale
#rsync -avz -e 'ssh' francaem@siam-linux20:TotRpwd  Lpwd 
