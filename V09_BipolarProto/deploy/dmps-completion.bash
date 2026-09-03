_dmps_complete()
{
    local commands="start stop restart status health log logs viewer update process ps url shell help"
    local service_commands="start stop restart status health log logs"
    if [[ "${COMP_WORDS[1]}" == "viewer" ]]; then
        COMPREPLY=($(compgen -W "${service_commands}" -- "${COMP_WORDS[2]}"))
    else
        COMPREPLY=($(compgen -W "${commands}" -- "${COMP_WORDS[1]}"))
    fi
}

complete -F _dmps_complete dmps
