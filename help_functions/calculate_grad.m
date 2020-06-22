function Term_T = calculate_grad(Term_T,T_ij,T_il,slack_new,slack_old)

%calculate (X_ij - X_il) corresponding to the triplet constraints
slack = slack_new - slack_old;
%remove from gradients for inactive constraints
Idx   = (slack==-1);
if ~isempty(Idx)
    T_12   = T_ij(Idx,:); T_13 = T_il(Idx,:);
    Term_T = Term_T - (T_12' * T_12 - T_13' * T_13);
end
%add to gradients for active constraints
Idx   = (slack==1);
if ~isempty(Idx)
    T_12   = T_ij(Idx,:); T_13 = T_il(Idx,:);
    Term_T = Term_T + (T_12' * T_12 - T_13' * T_13);
end

end
