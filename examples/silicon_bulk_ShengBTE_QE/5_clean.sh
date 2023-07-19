#!/bin/bash
printf "We\'d like to remove the following directories:\n"
printf "\t ${kaldo_outputs} ${kaldo_ald} 1x1x1 3x3x3 5x5x5 8x8x8\n"

printf "Type \"yes\" to continue ..\n"
read consent

if [ ${consent} == "yes" ];
then
    printf "Proceeding with cleanup\n"
    for dirs in ${kaldo_outputs} ${kaldo_ald} 1x1x1 3x3x3 5x5x5 8x8x8
    do
        printf "Removing ${dirs}\t"
        rm -r ${dirs}
    done
    printf "\n"
else
   printf "Consent revoked, exiting ..\n"
fi
