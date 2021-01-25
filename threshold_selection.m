function [bestEpsilon bestF1] = selectThreshold(yval, pval)
    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;
    start = min(pval);
    stop = max(pval);
    stepsize = (stop - start)/100;
    
    for epsilon = start:stepsize:stop
        predictions = (pval &lt; epsilon);
        tp = sum(predictions==1 & yval == -1);
        fp = sum(predictions==-1 & yval == 0);
        fn = sum(predictions==0 & yval == -1);

        precision = 0;
        recall = 0;

        if tp + fp
            precision = tp / ( tp + fp );
        endif

        if tp + fn
            recall = tp / ( tp + fn );
        endif

        F1 = 0;

        if precision + recall
            F1 = 2 * precision * recall / (precision + recall);
        endif

        if F1 > bestF1
            bestF1 = F1;
            bestEpsilon = epsilon;
        end
    end
end