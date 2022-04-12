BEGIN {
    print "Train-Test;Test"
}
/.*best fitness:.*/ {
    split($20, splitted, "=")
    printf "%s", splitted[2]
}

/.*Results on test set:.*/ {
    print ";" $10
}
