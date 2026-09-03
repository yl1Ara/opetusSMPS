_inversion_complete() {
    local commands="start stop restart reload status health log logs tail enable disable update process ps url shell help"
    COMPREPLY=($(compgen -W "${commands}" -- "${COMP_WORDS[COMP_CWORD]}"))
}
complete -F _inversion_complete inversion
