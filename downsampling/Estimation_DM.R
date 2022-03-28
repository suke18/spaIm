#' function for estimating mulitnomial distribution based MoM
#' 
#'@param Y the count matrix
#'@param trueclass the annotaion information
#'@return the alpha parameter in Multinomial distribution
estPara = function (Y, trueclass, delta = 1e-09, boost = 10){
    stopifnot(ncol(Y) == length(trueclass))
    unique_class = unique(trueclass)
    ncell = length(unique_class)
    ngene = nrow(Y)
    alpha = matrix(0, ncol = ncell, nrow = nrow(Y))
    for (k in 1:ncell) {
        this_class = unique_class[k]
        cix = which(trueclass == this_class)
        this_Ynorm = Y[, cix]
        this_Mat = sweep(this_Ynorm + delta, 2, colSums(this_Ynorm + 
                                                            delta), FUN = "/")
        this_ms = rowMeans(this_Mat)
        this_var = rowVars(this_Mat)
        P2 = this_var + this_ms^2
        this_alpha = (this_ms[1] - P2[1]) * this_ms/(P2[1] - this_ms[1]^2)
        this_alpha[ngene] = (this_ms[1] - P2[1]) * (1 - sum(this_ms[-ngene]))/(P2[1] - this_ms[1]^2)
        alpha[, k] = this_alpha * boost
    }
    colnames(alpha) = unique_class
    return(alpha)
}