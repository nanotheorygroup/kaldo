#!/bin/bash
printf "We\'d like to remove the following directories:\n"
printf "\t ${kaldo_outputs} ${kaldo_ald} ${kaldo_inputs}\n"

printf "Type \"yes\" to continue ..\n"
read consent

if [ ${consent} == "yes" ];
then
    printf "Proceeding with cleanup\n"
    for dirs in ${kaldo_outputs} ${kaldo_ald} ${kaldo_inputs}
    do
        printf "Removing ${dirs}\t"
        rm -r ${dirs}
    done
    printf "\n"
else
   printf "Consent revoked, exiting ..\n"
fi
