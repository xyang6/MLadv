function data = data_split(data)

data.Idx_train = cell(data.N_total,1);
data.Idx_val   = cell(data.N_total,1);
data.Idx_test  = cell(data.N_total,1);
for i = 1:data.N_total
    rng(i)
    [data.Idx_train{i,1},data.Idx_val{i},data.Idx_test{i}] = ...
        dividerand(data.N,data.prop_tr,data.prop_val,1-data.prop_tr-data.prop_val);
end;clear i

end