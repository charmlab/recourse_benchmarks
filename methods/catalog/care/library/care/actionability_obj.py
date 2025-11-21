def actionabilityObj(x_org, cf_org, user_preferences):

    constraint = user_preferences['constraint']
    importance = user_preferences['importance']

    cost = []
    idx =  [i for i, c in enumerate(constraint) if c is not None]
    for i in idx:
        if constraint[i] == 'fix':
            cost.append(int(cf_org[i] != x_org[i]) * importance[i])
        elif constraint[i] == 'l':
            cost.append(int(cf_org[i] >= x_org[i]) * importance[i])
        elif constraint[i] == 'g':
            cost.append(int(cf_org[i] <= x_org[i]) * importance[i])
        elif constraint[i] == 'ge':
            cost.append(int(cf_org[i] < x_org[i]) * importance[i])
        elif constraint[i] == 'le':
            cost.append(int(cf_org[i] > x_org[i]) * importance[i])
        elif type(constraint[i]) == set:
            cost.append(int(not(cf_org[i] in constraint[i])) * importance[i])
        elif type(constraint[i]) == list:
            cost.append(int(not(constraint[i][0] <= cf_org[i] <= constraint[i][1])) * importance[i])
        # is used for coherency validation experiment
        elif constraint[i] == 'change':
            cost.append(int(cf_org[i] == x_org[i]) * importance[i])
    return sum(cost)