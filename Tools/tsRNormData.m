function normData = tsRNormData(data,lowB,highB)
% Range normalization
    minData = min(data,[],1);
    maxData = max(data,[],1);    
    normData = bsxfun(@minus, data, minData);
    normData = bsxfun(@rdivide,normData,(maxData-minData));
    normData = bsxfun(@times,normData,(highB-lowB));
    normData = bsxfun(@plus,normData,lowB);

end