import logging, os

import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import confusion_matrix as CM
import sklearn.tree as T
from sklearn.base import clone
import math
from tqdm import tqdm
import copy

Feature_Data = []
Label_Data = []
rev = False

logging.getLogger().setLevel(logging.INFO)


# Transfer the feature values to defect_id list, feature name list and data list
# Input: label - feature values
# Output: Defect_id:List()
#         Feature name:List()
#         Data:List()
def convert2List(label, feature, test):
    data = []
    id_list = []
    result = {"detection": [], "misclassificaiton": [], "mis-type": [], "mis-type-strict": []}

    for i in label.keys():
        id_list.append(i)
        e = []
        for j in feature:
            e.append(label[i][j])
        data.append(e)
        result["detection"].append(int(test["detection"][i]))
        if int(test["detection"][i]) == 1:
            result["misclassificaiton"].append(0)
        else:
            result["misclassificaiton"].append(1)

        if int(test["type"][i]) == 1:
            result["mis-type"].append(0)
            result["mis-type-strict"].append(0)
        else:
            result["mis-type"].append(1)
            if int(test["detection"][i]) == 0:
                result["mis-type-strict"].append(0)
            else:
                result["mis-type-strict"].append(1)

    return id_list, feature, data, result


# Load feature data into model
# Input: Feature data
def load_feature_data(data):
    global Feature_Data
    Feature_Data = data


# Load Test data into model
# Input: Test data
def load_label_data(data):
    global Label_Data
    Label_Data = data


# Check all data is correctly loaded
def check_data():
    global Feature_Data, Label_Data
    if Feature_Data is None or len(Feature_Data) <= 0:
        logging.error("Feature_Data is missing or not correctly loaded")
    if Label_Data is None or len(Label_Data) <= 0:
        logging.error("Label_Data is missing or not correctly loaded")


# Plant trees - Training decision tree models
# Input: Feature name:List()
#        Number of decision tree:Integer - default:1
#        Criterion:"entropy" or "gini"
# Output: Trained model:sklearn
#         Evaluation matrix:[[tp,fp],[fn,tn]]
#         Error feature:[feature name]
def plant_trees(feature_name, n_tree=1, criterion="entropy", reverse=False):
    global Feature_Data, Label_Data, rev
    check_data()

    rev = reverse

    models = []
    evaluation = []
    errors = []
    dt = DTC(criterion=criterion, splitter="random", max_features=1, max_depth=None)
    for _ in tqdm(range(n_tree), desc="Plant trees"):
        dt_c = clone(dt)
        dt_m = dt_c.fit(Feature_Data, Label_Data)
        error = []
        predict = dt_m.predict(Feature_Data)
        for i in range(len(predict)):
            if predict[i] != Label_Data[i]:
                node_id = dt_m.apply([Feature_Data[i]])[0]
                if node_id in dt_m.tree_.children_left:
                    node_id = np.where(dt_m.tree_.children_left == node_id)
                elif node_id in dt_m.tree_.children_right:
                    node_id = np.where(dt_m.tree_.children_right == node_id)
                else:
                    continue
                wrong_feature = int(dt_m.tree_.feature[node_id])
                if feature_name[wrong_feature] not in error:
                    error.append(feature_name[wrong_feature])
        errors.append(error)
        evaluation.append(CM(Label_Data, predict))
        models.append(dt_m)
    return models, evaluation, errors


# Find all routes from the trees
# Input: Path:List() - Extracted path from trees
# Output: Routes:List()
def find_all_route(path):
    route = []
    temp = [[0, None]]
    for i in path:
        if i[0] == temp[-1][0]:
            temp[-1][-1] = True
            temp.append([i[1], None])

        else:
            route.append(copy.deepcopy(temp))
            temp = temp[:1 + temp.index([i[0], True])]
            temp[-1][-1] = False
            temp.append([i[1], None])

    route.append(copy.deepcopy(temp))

    return route


# Climb tree - Extract text-based tree to machine understandable arrays
# Input: Generated tree:graphviz
# Output: Path:List() - tree paths and node condition
#         Node:Dict() - tree nodes and relevant information
#         Route:List() - tree routes based on path
def climb_tree(tree):
    tree1 = tree.split("\n")[:-1]
    path = []
    node = {}
    for i in tree1:
        try:
            int(i[0])
        except:
            continue
        if "->" in i:
            p = i.split(";")[0].split("[")[0].split("->")
            path.append((int(p[0]), int(p[1])))
        else:
            node_i = int(i.split('"')[0].split(" ")[0])
            n = i.split('"')[1].split("\\n")
            info = {}
            if n[0].startswith("gini") or n[0].startswith("entropy"):
                info["leaf"] = True
                info["criterion"] = float(n[0].split("=")[-1])
                info["samples"] = int(n[1].split("=")[-1])
                value = [int(float(n[2].split("=")[-1].split(",")[0][2:]))]
                for j in range(1, len(n[2].split("=")[-1].split(",")) - 1):
                    value.append(int(n[2].split("=")[-1].split(",")[j][1:]))
                value.append(int(float(n[2].split("=")[-1].split(",")[-1][:-1])))
                info["value"] = value
                info["score"] = -1
            else:
                info["leaf"] = False

                info["feature"] = n[0].split(" ")[0]
                info["threshold"] = float(n[0].split(" ")[-1])

                info["criterion"] = float(n[1].split("=")[-1])
                info["samples"] = int(n[2].split("=")[-1])

                value = [int(n[3].split("=")[-1].split(",")[0][2:])]
                for j in range(1, len(n[3].split("=")[-1].split(",")) - 1):
                    value.append(int(n[3].split("=")[-1].split(",")[j][1:]))
                value.append(int(n[3].split("=")[-1].split(",")[-1][:-1]))
                info["value"] = value
                info["score"] = -1

            node[node_i] = info

    route = find_all_route(path)

    for rs in route:
        depth = 0
        for r in rs:
            if "depth" not in node[r[0]]:
                node[r[0]]["depth"] = depth
            depth += 1

    return path, node, route


# Climb tree (Forest) - Extract text-based tree to machine understandable arrays
# Input: Generated mode - trained sklearn models
#        Feature name - feature name list
# Output: Path:List() - tree paths and node condition
#         Node:Dict() - tree nodes and relevant information
#         Route:List() - tree routes based on path
def climb_trees(model, feature_name):
    routes = []
    nodes = []
    paths = []
    for i in tqdm(model, desc="Climb trees"):
        p, n, r = climb_tree(T.export_graphviz(i, feature_names=feature_name))
        routes.append(r)
        nodes.append(n)
        paths.append(p)
    return paths, nodes, routes


# Validate Trees - are the trained trees satisfied to make analysis
# Input: Evaluation matrix
#        Pass rate:Float[0-1] - Evaluation score lowest bound
#        Class Importance:[1,1]- class weight
# Output: Validation:Boolean - is tree satisfied to continue
#         Score list:List() - good score, T score and range, F score and range
def val_trees(tree_eval, pass_rate=0.95, class_importance=None):
    global rev
    if class_importance is None:
        class_importance = [1 for _ in tree_eval[0][0]]
    else:
        if len(class_importance) != len(tree_eval[0][0]):
            raise ValueError(
                "Class importance does not match with test label. Expecting: {0}".format(len(tree_eval[0][0])))
    t0_prec = []
    t1_prec = []
    good = 0
    for i in tqdm(tree_eval, desc="Validate trees"):
        t0_pr = 0
        if i[0][0] == 0:
            t0_prec.append(0.0)
        else:
            t0_pr = i[0][0] / sum(i[0])
            t0_prec.append(t0_pr)
        t1_pr = 0
        if i[-1][-1] == 0:
            t1_prec.append(0.0)
        else:
            t1_pr = i[1][-1] / sum(i[1])
            t1_prec.append(t1_pr)

        good += t0_pr * class_importance[0] / sum(class_importance)
        good += t1_pr * class_importance[1] / sum(class_importance)

    t0_avg = sum(t0_prec) / len(t0_prec)
    t1_avg = sum(t1_prec) / len(t1_prec)
    t0_range = max(t0_prec) - min(t0_prec)
    t1_range = max(t1_prec) - min(t1_prec)
    good_rate = good / len(tree_eval)

    if rev:
        logging.info("The report analyses the result=0 rules")
    else:
        logging.info("The report analyses the result=1 rules")
    logging.info("{0}% rules of result=0 are learned.".format(round(t0_avg * 100, 2)))
    logging.info("{0}% rules of result=1 are learned.".format(round(t1_avg * 100, 2)))
    if good_rate >= pass_rate:
        logging.info("Trees are validated and pass")
        return True, [good_rate, (t0_avg, t0_range), (t1_avg, t1_range)]
    else:
        logging.warning(
            "Validation rate is lower than the pass requirements. Required: {0};  Current: {1}".format(
                pass_rate, round(good_rate, 3)))
        return False, [good_rate, (t0_avg, t0_range), (t1_avg, t1_range)]


# Find the node's children node ids
# Input: node:id
#        node list:Dict()
#        Path:List()
# output: Children:[c1, c2] - if leaf, then [None, None]
def find_children(n, no, p):
    if no[n]["leaf"]:
        return None, None
    else:
        children = list(filter(lambda x: x[0] == n, p))
        return children[0][-1], children[-1][-1]


# Determine the distinguish-ability status
# Input: t0 - True child distinguish score
#        t1 - False child distinguish score
# Output: level_text:String - Confirmation, reduction, half reduction
def determine_level(t0, t1):
    if (t0 == 0.0 and t1 == 1.0) or (t0 == 1.0 and t1 == 0.0):
        return "confirmation"
    elif t0 == t1:
        return "reduction"
    elif t0 == 1.0 or t0 == 0.0 or t1 == 0.0 or t1 == 1.0:
        return "confirmation"
    elif t0 == 0.5 or t1 == 0.5:
        return "half reduction"
    else:
        return "reduction"


# Calculate decision and distinguish scores
# Input: node value - mis-classification node values
#        true value - mis-classification true child values
#        false value - mis-classification false child values
# Output: Decision score
#         Distinguish score
#         (direction to true, direction to f)
#         (direction to misclassified, direction to classified)
#         decision to mis-classification
#         decision status
def tree_decision(n, t, f):
    global rev
    t0 = t[0] / n[0]
    t1 = t[-1] / n[-1]
    f0 = f[0] / n[0]
    f1 = f[-1] / n[-1]

    decision = (abs(t0 - f0) + abs(t1 - f1)) / 2
    distinguish = abs(t1 - t0)
    direction_t = (t[-1] - t[0]) / (t[0] + t[-1])
    direction_f = (f[-1] - f[0]) / (f[0] + f[-1])
    if rev:
        direction_t = (t[0] - t[-1]) / (t[0] + t[-1])
        direction_f = (f[0] - f[-1]) / (f[0] + f[-1])

    direction_0 = t0 - f0
    direction_1 = t1 - f1
    decision_tf = direction_t >= direction_f

    decision_status = determine_level(t0, t1)

    return decision, distinguish, (direction_t, direction_f), (direction_0, direction_1), decision_tf, decision_status


# Determine the distinguish degree
# Input: Value - Distinguish score
#        Threshold - boundary for full, strong, middle, weak and empty
# Output: Degree of distinguish:String - full, strong, middle, weak and empty
def degree_status(value, threshold):
    if value <= 0.0:
        return "empty"
    elif value >= 1.0:
        return "full"
    elif value < threshold[0]:
        return "weak"
    elif value < threshold[1]:
        return "middle"
    else:
        return "strong"


# Analyse trees and calculate scores
# Input: Path
#        Node
#        Error features
#        Feature name list
#        Round number:Integer - if 0 then no round, else round to the closest decimal value
#        degree_threshold - distinguish degree boundary
#        own_result
# Output: Result:Dict(): for each tree for each feature and result
def analyse_trees(path, node, error, feature, round_num=0, degree_threshold=(0.25, 0.5), own_result=None):
    global rev

    tree_result = []
    if own_result is not None:
        if len(node) != len(own_result):
            raise ValueError("Own results do not match the generated trees structure.\n There is/are {0} trees, "
                             "but get {1} trees in own result".format(len(node), len(own_result)))

    for i in tqdm(range(len(node)), desc="Analyse trees"):
        total = sum(node[i][0]["value"])
        feature_result = {}
        for f in feature:
            feature_result[f] = []
        for j in node[i].keys():
            t, n = find_children(j, node[i], path[i])

            if t is None:
                continue

            value = node[i][j]["value"]
            true = node[i][t]["value"]
            false = node[i][n]["value"]
            decision, distinguish, direction_tf, direction_01, decision_tf, decision_status = \
                tree_decision(value, true, false)
            usage = sum(value) / total
            status_degree = degree_status(distinguish, degree_threshold)

            criterion = node[i][j]["criterion"]
            if round_num > 0:
                distinguish = round(distinguish, round_num)
                decision = round(decision, round_num)
                direction_tf = (round(direction_tf[0], round_num), round(direction_tf[1], round_num))
                direction_01 = (round(direction_01[0], round_num), round(direction_01[1], round_num))
                usage = round(usage, round_num)
            threshold = node[i][j]["threshold"]
            fea = node[i][j]["feature"]
            mistake = fea in error[i]
            result = {"tree": i,
                      "node": j,
                      "threshold": threshold,
                      "usage": usage,
                      "decision": decision,
                      "distinguish": distinguish,
                      "direction_tf": direction_tf,
                      "direction_01": direction_01,
                      "target_tf": decision_tf,
                      "status": decision_status,
                      "status_degree": status_degree,
                      "mistake": mistake}

            if own_result is not None:
                for result_name in own_result[i][j].keys():
                    if result_name in result:
                        raise KeyError("Result name duplicated: {0}".format(result_name))
                    else:
                        result[result_name] = own_result[i][j][result_name]

            feature_result[fea].append(result)
        tree_result.append(feature_result)

    return tree_result


# Calculate the rank of the feature based on the overall score
# Input: Feature score list:List()
# Output: Rank list:List() - the rank for each feature
def ranking(l):
    ranked = []
    rank = 0
    sort_l = list(sorted(copy.deepcopy(l), reverse=True))
    temp = {}
    for i in sort_l:
        if i not in temp:
            rank += 1
        temp[i] = rank

    for i in l:
        ranked.append(temp[i])

    return ranked


# Analyse threshold boundaries of each feature to locate mis-classification prediction based on the highest scores in one tree
# Input: Result list
#        Feature range - the min and max value of this feature
# Output: Lower bound and Upper bound
def threshold_split_1(result, feature_range):
    global rev
    lower = []
    upper = []
    max_b = max(feature_range)
    min_b = min(feature_range)
    re_t = sorted(list(filter(lambda x: x["target_tf"], result)), key=lambda x: x["score"], reverse=True)
    re_f = sorted(list(filter(lambda x: not x["target_tf"], result)), key=lambda x: x["score"], reverse=True)

    for i in re_t:
        if rev:
            if i["direction_01"][0] * i["distinguish"] * i["usage"] < 0:
                upper.append((i["threshold"], abs(i["direction_01"][0] * i["distinguish"] * i["usage"]) / 2))
            else:
                upper.append((i["threshold"], i["direction_01"][0] * i["distinguish"] * i["usage"]))
        else:
            if i["direction_01"][-1] * i["distinguish"] * i["usage"] < 0:
                upper.append((i["threshold"], abs(i["direction_01"][-1] * i["distinguish"] * i["usage"]) / 2))
            else:
                upper.append((i["threshold"], i["direction_01"][-1] * i["distinguish"] * i["usage"]))

    for i in re_f:
        if rev:
            if (0 - i["direction_01"][0]) * i["distinguish"] * i["usage"] < 0:
                lower.append((i["threshold"], abs((0 - i["direction_01"][0]) * i["distinguish"] * i["usage"]) / 2))
            else:
                lower.append((i["threshold"], (0 - i["direction_01"][0]) * i["distinguish"] * i["usage"]))
        else:
            if (0 - i["direction_01"][-1]) * i["distinguish"] * i["usage"] < 0:
                lower.append((i["threshold"], abs((0 - i["direction_01"][-1]) * i["distinguish"] * i["usage"]) / 2))
            else:
                lower.append((i["threshold"], (0 - i["direction_01"][-1]) * i["distinguish"] * i["usage"]))

    lower = sorted(lower, key=lambda x: x[-1], reverse=True)
    upper = sorted(upper, key=lambda x: x[-1], reverse=True)
    if len(lower) == 0 and len(upper) == 0:
        return (min_b, 0), (max_b, 0)
    elif len(lower) == 0:
        return (min_b, upper[0][1]), upper[0]
    elif len(upper) == 0:
        return lower[0], (max_b, lower[0][1])
    else:
        return lower[0], upper[0]


# Analyse the range/boundaries of the mis-classification of each feature in all trees
# Input: Lower bound and Upper bound
#        top_n:Integer - only calculate the top n scores
#        Max move: max threshold moving bound
# Output: Lower bound and Upper bound
#         Range score:Float - range moving gap between the initial and final range
#         Range stability:[Flat, Float] - the stability of range is changed every iteration
def bound_range(lower, upper, top_n, max_move=1):
    r = [lower[0][0], upper[0][0]]
    s = [lower[0][1], upper[0][1]]
    lower_stable = 0
    upper_stable = 0
    for i in range(top_n):
        lower_rate = (1 - (s[0] - lower[i][1]) / s[0]) / top_n
        if r[0] <= 0:
            if (lower[i][0] - r[0]) > 0:
                lower_gap = (lower[i][0] - r[0])
            else:
                lower_gap = 0
        else:
            lower_gap = (lower[i][0] - r[0]) / r[0]
        upper_rate = (1 - (s[1] - upper[i][1]) / s[1]) / top_n
        upper_gap = (upper[i][0] - r[1]) / r[1]
        lower_stable += (lower[i][0] - r[0]) / top_n
        upper_stable += (upper[i][0] - r[1]) / top_n
        if r[0] <= 0:
            if abs(lower_gap) > max_move:
                if lower_gap > 0:
                    lower_gap = max_move
                else:
                    lower_gap = 0
        else:
            if abs(lower_gap) > max_move / r[0]:
                if lower_gap > 0:
                    lower_gap = max_move / r[0]
                else:
                    lower_gap = 0 - max_move / r[0]

        if abs(upper_gap) > max_move / r[1]:
            if upper_gap > 0:
                upper_gap = max_move / r[1]
            else:
                upper_gap = 0 - max_move / r[1]

        r[0] += r[0] * lower_gap * lower_rate
        r[1] += r[1] * upper_gap * upper_rate

    range_score = 0
    if (r[0] + lower[0][0]) <= 0:
        range_score = abs(r[0] - lower[0][0])
    else:
        range_score = abs(r[0] - lower[0][0]) / (r[0] + lower[0][0])
    range_score += abs(r[1] - upper[0][0]) / (r[1] + upper[0][0])
    range_score /= 2
    return r, range_score, [lower_stable, upper_stable]




# Calculate the overall score based on count and score - based on feature
# Input: Count and score list:List((count, score))
#        Count Score weight
# Output: Overall score:Float
def c_s_overall_feature(cs_list, c_s):
    overall = 0
    for i in cs_list:
        count = 1
        if (max([j[0] for j in cs_list]) - min([j[0] for j in cs_list])) > 0:
            count = (i[0] - min([j[0] for j in cs_list])) / (
                        max([j[0] for j in cs_list]) - min([j[0] for j in cs_list]))
        overall += count * c_s[0] + i[1] * c_s[1]
    overall /= len(cs_list)
    return overall

# Calculate the overall score based on count and score
# Input: Count and score list:List((count, score))
#        Count Score weight
# Output: Overall score:Float
def c_s_overall(count, score, c_s):

    return count * c_s[0] + score * c_s[1]


# Summary the analysed tree and calculate the final score and routes for all features and all trees
# Input: Analysed tree
#        Feature name
#        Routes
#        Nodes
#        Feature range
#        Feature weight:List() - default: "balance" or a list of integers and length must equal to feature name
#        Status bonus:List() - default:[0.2,0.5] - if the decision is confirmation or half reduction, score will get a bonus
#        Degree bonus:List() - default:[0,0.1,0.2,0.3,0.5] - if the decision is empty,weak,middle,strong,full, score will get a bonus
#        Mistake punish:Float() - default:2.0 - if decision make mistakes, the score will be reduced
#        All positive:Boolean - the score must be positive?
#        Own result
#        Top n:Integer - default:10 - pick top n score as range boundary
#        Count Score weight
# Output: Report:Dict() - report contain the score and relevant information need to write
#         Mis-classification routes
#         Classification routes
#         Node
def summary_trees(tree_result, feature, route, node, feature_range, feature_weight="balance", status_bonus=None,
                  degree_bonus=None, mistake_punish=2, all_positive=True, own_result=None, top_n=10, c_s=None):
    if c_s is None:
        c_s = [0.25, 0.75]
    if status_bonus is None:
        status_bonus = [0.2, 0.5]
    else:
        if len(status_bonus) != 5:
            raise ValueError("Status bonus must have 2 values in the list")

    if degree_bonus is None:
        degree_bonus = [0, 0.1, 0.2, 0.3, 0.5]
    else:
        if len(degree_bonus) != 5:
            raise ValueError("Degree bonus must have 5 values in the list")
    weight = [1 for i in range(len(feature))]

    report = {}
    for f in feature:
        report[f] = {"feature": f,
                     "count": 0,
                     "score": 0,
                     "range": ([], []),
                     "overall": []}

    if feature_weight != "balance":
        if len(feature_weight) == len(feature):
            weight = feature_weight
        else:
            raise ValueError("feature weight list does not match the feature list")
    weight = [i / max(weight) for i in weight]
    for tree in tqdm(range(len(tree_result)), desc="Summarise trees - Brief Report"):
        fi = 0
        for f in feature:
            for fr in tree_result[tree][f]:

                score = fr["distinguish"]
                score *= 1 + fr["decision"]

                if fr["status"] == "half reduction":
                    score += status_bonus[0]
                elif fr["status"] == "confirmation":
                    score += status_bonus[1]

                if fr["status_degree"] == "empty":
                    score += degree_bonus[0]
                elif fr["status_degree"] == "weak":
                    score += degree_bonus[1]
                elif fr["status_degree"] == "middle":
                    score += degree_bonus[2]
                elif fr["status_degree"] == "strong":
                    score += degree_bonus[3]
                elif fr["status_degree"] == "full":
                    score += degree_bonus[4]

                if fr["mistake"]:
                    score -= mistake_punish

                if own_result is not None:
                    for i in own_result:
                        score += fr[i]

                if score < 0 and all_positive:
                    score = 0

                score *= fr["usage"]
                fr["score"] = score * weight[fi]
                node[fr["tree"]][fr["node"]]["score"] = score * weight[fi]

            scores = [i["score"] for i in tree_result[tree][f]]
            report[f]["count"] += len(scores) / len(tree_result)
            score_avg = 0
            if len(scores) == 0:
                report[f]["score"] += 0
            else:
                score_avg = sum(scores) / len(scores)
                report[f]["score"] += score_avg / len(tree_result)

            # report[f]["overall"].append([len(scores), score_avg])
            lower, upper = threshold_split_1(tree_result[tree][f], feature_range[f])
            report[f]["range"][0].append(lower)
            report[f]["range"][1].append(upper)
            report[f]["range"] = (sorted(report[f]["range"][0], key=lambda x: x[-1], reverse=True),
                                  sorted(report[f]["range"][1], key=lambda x: x[-1], reverse=True))

            fi += 1

    for f in report.keys():
        if top_n == 0:
            top_n = len(report[f]["range"][0])

        if top_n > len(report[f]["range"][0]):
            logging.warning("top_n value is larger than the number of trees. top_n: {0}, Trees: {1}\n\t"
                            "Replace the size of tree to the top_n".format(top_n, len(report[f]["range"][0])))
            top_n = len(report[f]["range"][0])
        report[f]["range"], report[f]["range_score"], report[f]["range_stable"] = bound_range(report[f]["range"][0],
                                                                                              report[f]["range"][1],
                                                                                              top_n)
        # report[f]["overall"] = c_s_overall(report[f]["overall"], c_s)

    for f in report.keys():
        report[f]["count_norm"] = (report[f]["count"] - min([report[i]["count"] for i in report.keys()])) / (
                max([report[i]["count"] for i in report.keys()]) - min(
            [report[i]["count"] for i in report.keys()]))
        report[f]["score_norm"] = (report[f]["score"] - min([report[i]["score"] for i in report.keys()])) / (
                max([report[i]["score"] for i in report.keys()]) - min(
            [report[i]["score"] for i in report.keys()]))

    for f in report.keys():
        report[f]["overall"] = c_s_overall(report[f]["count_norm"], report[f]["score_norm"], c_s)

    for f in report.keys():
        report[f]["overall_norm"] = (report[f]["overall"] - min([report[i]["overall"] for i in report.keys()])) / (
                max([report[i]["overall"] for i in report.keys()]) - min(
            [report[i]["overall"] for i in report.keys()]))

    score_ranks = ranking([report[i]["overall"] for i in report.keys()])
    for i in range(len(score_ranks)):
        report[list(report.keys())[i]]["rank"] = score_ranks[i]

    route_1 = []
    route_0 = []
    for tree in tqdm(range(len(tree_result)), desc="Summarise trees - Route Report"):

        for r in route[tree]:
            score_r = [node[tree][i[0]]["score"] for i in r[:-1]]
            feature_r = [node[tree][i[0]]["feature"] for i in r[:-1]]
            threshold_r = [node[tree][i[0]]["threshold"] for i in r[:-1]]
            tf_r = [i[1] for i in r[:-1]]
            overall_r = sum(score_r) / len(score_r)
            decision_r = node[tree][r[-1][0]]["value"][1] >= node[tree][r[-1][0]]["value"][0]

            if decision_r:
                decision_r = 1
            else:
                decision_r = 0
            value_r = node[tree][r[-1][0]]["value"]
            length_r = len(r) - 1

            result_dir = {"overall": overall_r,
                          "decision": decision_r,
                          "value": value_r,
                          "length": length_r,
                          "score": score_r,
                          "feature": feature_r,
                          "threshold": threshold_r,
                          "tf": tf_r,
                          }
            if decision_r == 1:
                route_1.append(result_dir)
            else:
                route_0.append(result_dir)

    return report, route_1, route_0, node


# Convert range to text
# Input: Feature range
#        Range
# Output: Text:String
def text_range(fr, r):
    rl = copy.deepcopy(fr)
    text = "<"
    count = 0
    for i in r:
        if i[0] < min(fr):
            rl.insert(0, None)
        elif i[0] > max(fr):
            rl.insert(len(fr), None)
        else:
            rl.insert(rl.index(i[0]), None)
        if i[1] > max(fr):
            rl.insert(len(fr), None)
        elif i[1] < min(fr):
            rl.insert(1, None)
        else:
            rl.insert(rl.index(i[1]) + 1, None)
    rl_clean = [rl[0]]

    for i in range(1, len(rl)):
        if rl[i] is None and rl[i - 1] is None:
            rl_clean = rl_clean[:-1]
        else:
            rl_clean.append(rl[i])

    start = True
    for i in rl_clean:
        if i is None:
            if start:
                text += "["
                start = not start
            else:
                text += "]"
                start = not start
        else:
            text += " " + str(i) + " "

    text += ">"
    return text


# Write route
# Input: Feature list
#        Threshold list
#        Decision list
# Output: Text:String
def write_route(feature, threshold, tf):
    text = ""
    for i in range(len(feature)):
        text += feature[i] + " "
        if tf[i]:
            text += "<="
        else:
            text += ">="
        text += " " + str(threshold[i]) + " "
        if i < len(feature) - 1:
            text += "and "
    return text


# Convert the report to text-based string
# Input: Report
#        Mis-classification routes
#        Classification routes
#        Nodes
#        Feature range
#        Top routes:Integer - default:5 - print the top n routes based on the determining values
#        Stable move:Float - default:0.75 - Drawing a flexible range with discount/bonus move based on the stability
#        Save - save the text into a txt file
# Output: Text-based report:String
def explain_tree(report, route_1, route_0, node, feature_range, route_top=5, stable_move=0.75, save=None):
    global rev
    output_text = "Forest Monkey Report\n\n"
    output_text += "===Summary Report===\n"

    output_text += "The summary section will show the importance of each feature.\n" \
                   "The rank is according to its count and score values.\n" \
                   "The features in higher ranks are relatively stronger, but not definitely, to explain " \
                   "the mis-classification reasons than others.\n" \
                   "Each feature will relate with its corresponding improvement solution, and all solutions are list " \
                   "at end of this report.\n------------\n" \
                   "Rank: the rank of the feature distinguish-ability\n---\n" \
                   "Overall Score: the overall score of the feature distinguish-ability.\n" \
                   "The higher value the better.\n---\n" \
                   "Count: the number of the average occurrence which this feature is used in all trees\n" \
                   "The higher number presents that this feature is popular and partially important for " \
                   "distinguishing the mis-classification\n---\n" \
                   "Importance Score: the average score of the decision importance by this feature in all trees\n" \
                   "The higher value presents that this feature can distinguish the mis-classification better\n---\n" \
                   "Range: the approximate range of this feature might cause mis-classification\n---\n" \
                   "Range Stability: the average stability of the top_n ranges in each feature\n" \
                   "The lower value presents that the range is stable and the mis-classification is more accurate\n" \
                   "---\n" \
                   "Range Difference: the gap distance between the initial range and the final range\n" \
                   "The lower value the better\n---\n" \
                   "Description: a short text to describe the mis-classification situation with range visualisation\n" \
                   "---\n" \
                   "* Please refer the range and value to determine the reasons why the model make prediction to " \
                   "achieve the target\n=============\n\n"
    for i in sorted(report.values(), key=lambda x: x["rank"]):
        left = i["range"][0]
        right = i["range"][1]
        ls = i["range_stable"][0]
        rs = i["range_stable"][1]
        fr = feature_range[i["feature"]]
        desc = "The result might be 1 happened when "
        if rev:
            desc = "The result might be 0 (true target) happened when "
        if round(left, 1) > round(right, 1):
            if (right + rs * stable_move) - (left + ls * stable_move) >= 0:
                if round(left + ls * stable_move) == round(right + rs * stable_move):
                    desc += "{f} value is around {v}, but, might contain mistakes. Range Visualisation: ".format(
                        f=i["feature"], v=round(left + ls * stable_move))
                    desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
                else:
                    if round(left + ls * stable_move) <= round(right + rs * stable_move):
                        desc += "{f} value is around between {v1} and {v2}, but, might contain mistakes. Range " \
                                "Visualisation: ".format(f=i["feature"], v1=round(left + ls * stable_move),
                                                         v2=round(right + rs * stable_move))
                        desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
                    else:
                        desc += "{f} value is around between {v1} and {v2}, but, might contain mistakes. Range " \
                                "Visualisation: ".format(f=i["feature"], v2=round(left + ls * stable_move),
                                                         v1=round(right + rs * stable_move))
                        desc += text_range(fr, [[round(right + rs * stable_move), round(left + ls * stable_move)]])
            else:
                if int(right + rs * stable_move) == int(left + ls * stable_move) + 1:
                    if (left + ls * stable_move) - (right + rs * stable_move) < 0.5:
                        desc += "{f} value is any possible value between all its range. Range Visualisation: ".format(
                            f=i["feature"])
                    else:
                        desc += "{f} value is any possible value between all its range, but might not include {v1}. " \
                                "Range Visualisation: ".format(f=i["feature"], v1=int(right + rs * stable_move))
                    desc += text_range(fr, [[min(fr), int(right + rs * stable_move)],
                                            [int(left + ls * stable_move) + 1, max(fr)]])
                else:
                    if int(right + rs * stable_move) == int(left + ls * stable_move):
                        desc += "{f} value is any possible value between all its range, but might not include {v1}. " \
                                "Range Visualisation: ".format(f=i["feature"], v1=int(right + rs * stable_move))
                    else:
                        desc += "{f} value <= {v1} or >= {v2}. Range Visualisation: ".format(
                            f=i["feature"], v1=int(right + rs * stable_move), v2=int(left + ls * stable_move) + 1)
                    desc += text_range(fr, [[min(fr), int(right + rs * stable_move)],
                                            [int(left + ls * stable_move) + 1, max(fr)]])
        else:
            if round(left + ls * stable_move) == round(right + rs * stable_move):
                desc += "{f} value is around {v}. Range Visualisation: ".format(f=i["feature"],
                                                                                v=round(left + ls * stable_move))
                desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
            else:
                if round(left + ls * stable_move) <= round(right + rs * stable_move):
                    desc += "{f} value is around between {v1} and {v2}. Range Visualisation: ".format(
                        f=i["feature"], v1=round(left + ls * stable_move), v2=round(right + rs * stable_move))
                    desc += text_range(fr, [[round(left + ls * stable_move), round(right + rs * stable_move)]])
                else:
                    desc += "{f} value is around between {v1} and {v2}, but might contain mistakes. Range " \
                            "Visualisation: ".format(f=i["feature"], v2=round(left + ls * stable_move),
                                                     v1=round(right + rs * stable_move))

                    desc += text_range(fr, [[round(right + rs * stable_move), round(left + ls * stable_move)]])

        if ls >= 0.5 and rs < 0.5:
            desc += "\nBut the lower boundary might be further lower/higher."
        elif ls < 0.5 and rs >= 0.5:
            desc += "\nBut the upper boundary might be further lower/higher."
        elif ls >= 0.5 and rs >= 0.5:
            desc += "\nBut the both boundaries might be further lower or higher."

        output_text += "Rank: {rank} - Feature: {feature} - Overall Score: {overall_norm} ({overall}) " \
                       "- Count: {count_norm} ({count}) - Importance Score: {score_norm} ({score}) - Range: {rang} " \
                       "- Range Stability: {rang_stable} - Range Difference: {ranges} \nDescription: {desc}" \
                       "\n---------\n".format(
            rank=i["rank"],
            feature=i["feature"],
            overall=round(i["overall"] * 100, 2),
            overall_norm=round(i["overall_norm"] * 100, 2),
            count=round(i["count"], 1),
            count_norm=round(i["count_norm"] * 100, 2),
            score=round(i["score"] * 100, 2),
            score_norm=round(i["score_norm"] * 100, 2),
            rang=[round(i["range"][0], 1),
                  round(i["range"][1], 1)],
            ranges=round(i["range_score"], 2), desc=desc,
            rang_stable=[round(i["range_stable"][0], 1),
                         round(i["range_stable"][1], 1)])

    output_text += "\n===Route Report===\n"
    output_text += "Route report section will show the feature conditions which might cause mis-classification " \
                   "or correct classification\n" \
                   "The score of each route is calculated based on the score of each feature and " \
                   "the decision routes in all trees.\n" \
                   "This section will select top meaningful routes which are based on the route score, route length " \
                   "and route decision values.\n" \
                   "The report contains the route and its related overall score, length, final values and " \
                   "mis-classification decision\n-------------\n" \
                   "Overall Score: the average score of this route\n" \
                   "Decision: the decision of the mis-classificaiton on this route\n" \
                   "Value: the number of defects is classified by following this route\n" \
                   "Length: the route length\n=============\n"
    route_1_overall = sorted(route_1, key=lambda x: x["overall"], reverse=True)
    route_0_overall = sorted(route_0, key=lambda x: x["overall"], reverse=True)
    output_text += "\n===Route Score Rank===\n"
    count = 1
    output_text += "\n-----Result to 1-----\n"
    for i in route_1_overall[:route_top]:
        output_text += "Rank: {rank} - Overall Score: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][0], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    count = 1
    if rev:
        output_text += "\n-----Result to 0 (true target)-----\n"
    else:
        output_text += "\n-----Result to 0-----\n"
    for i in route_0_overall[:route_top]:
        output_text += "Rank: {rank} - Overall Score: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][1], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    output_text += "\n===Route Length===\n"
    count = 1
    output_text += "\n-----Result to 1-----\n"
    route_1_len = sorted(route_1, key=lambda x: x["length"], reverse=False)
    route_0_len = sorted(route_0, key=lambda x: x["length"], reverse=False)
    for i in route_1_len[:route_top]:
        output_text += "Rank: {rank} - Overall Score: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][0], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    count = 1
    if rev:
        output_text += "\n-----Result to 0 (true target)-----\n"
    else:
        output_text += "\n-----Result to 0-----\n"
    for i in route_0_len[:route_top]:
        output_text += "Rank: {rank} - Overall Score: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][1], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    output_text += "\n===Route Value===\n"
    route_1_value = sorted(route_1, key=lambda x: x["value"][1], reverse=True)
    route_0_value = sorted(route_0, key=lambda x: x["value"][0], reverse=True)
    count = 1
    output_text += "\n-----Result to 1-----\n"
    for i in route_1_value[:route_top]:
        output_text += "Rank: {rank} - Overall Score: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][0], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1
    count = 1
    if rev:
        output_text += "\n-----Result to 0 (true target)-----\n"
    else:
        output_text += "\n-----Result to 0-----\n"
    for i in route_0_value[:route_top]:
        output_text += "Rank: {rank} - Overall Score: {overall} - Result?: {decision} - Value: {value} - " \
                       "Length: {length}\nRoute: ".format(rank=count, overall=i["overall"], decision=i["decision"],
                                                          value=i["value"][1], length=i["length"])
        output_text += write_route(i["feature"], i["threshold"], i["tf"]) + "\n"
        output_text += "------------------\n"
        count += 1

    output_text += "\n===Improvement Explanation===\n"
    output_text += "This section will explain the possible processes for improving the detection " \
                   "performance. However, the possible improvement solutions are flexible and not limited.\n" \
                   "Generally, increasing dataset will improve the detection performance\n----------\n"
    output_text += "Size:\n1. Increase dataset amount of the corresponding sized defect\n2. Enlarge the defect area\n"
    output_text += "Coverage:\n1. Improve detection model architecture\n2. Suitable image augmentations\n"
    output_text += "Aspect Ratio:\n1. Improve detection model architecture\n2. Suitable image augmentations\n" \
                   "3. Normalise aspect ratio\n"
    output_text += "Average Vertex Degree:\n1. Improve detection model architecture\n2. Increase image features\n"
    output_text += "Mode Vertex Degree:\n1. Improve detection model architecture\n2. Increase image features\n"
    output_text += "Number of Edge:\n1. Suitable image augmentations\n2. Improve detection model architecture\n"
    output_text += "Average Edge Length:\n1. Suitable image augmentations\n2. Improve detection model architecture\n"
    output_text += "Mode Edge Length:\n1. Suitable image augmentations\n2. Improve detection model architecture\n"
    output_text += "Neighbour Distance:\n1. Separate neighbour defect to individual image if mis-classification " \
                   "happen with closed neighbour\n2. Increase dataset of similar situation\n"
    output_text += "Shape Complexity (Edge Ratio, Follow Turn, Small Turn, Reverse Turn):\n" \
                   "1. Suitable image augmentations\n" \
                   "2. Improve detection model architecture\n" \
                   "3. Increase image features\n" \
                   "4. Increase dataset\n"
    output_text += "Average HUE:\n1. Increase dataset\n"
    output_text += "Mode HUE:\n1. Increase dataset\n"
    output_text += "HUE Range:\n1. Normalise colour\n2. Improve detection model architecture\n"
    output_text += "Unique HUE:\n1. Normalise colour\n2. Improve detection model architecture\n3. Grey-scale image " \
                   "if value is large\n"
    output_text += "Average Saturation:\n1. Increase dataset\n2. Grey-scale image if value is large\n"
    output_text += "Mode Saturation:\n1. Increase dataset\n2. Grey-scale image if value is large\n"
    output_text += "Saturation Range:\n1. Normalise colour\n2. Grey-scale image if value is large\n"
    output_text += "Unique Saturation:\nn1. Normalise colour\n2. Improve detection model architecture\n3. Grey-scale " \
                   "image if value is large\n"
    output_text += "Average Brightness:\n1. Increase dataset\n2. Suitable image pre-processing\n"
    output_text += "Mode Brightness:\n1. Increase dataset\n2. Suitable image pre-processing\n"
    output_text += "Brightness Range:\n1. Normalise colour\n"
    output_text += "Unique Brightness:\n1. Normalise colour\n"
    output_text += "Colour Complexity (HUE, Saturation, and Brightness):\n1. Normalise histogram of the image\n" \
                   "2. Improve detection model architecture\n"
    output_text += "Mode Hue (Outside):\n1. Normalise colour\n2. Increase dataset\n3. Improve detection model " \
                   "architecture\n "
    output_text += "Mode Saturation (Outside):\n1. Normalise colour\n2. Increase dataset\n3. Adjust contrast or " \
                   "histogram\n"
    output_text += "Mode Brightness (Outside):\n1. Normalise colour\n2. Increase dataset\n3. Adjust contrast or " \
                   "histogram\n"

    if save is not None:
        if save.endswith(".txt"):
            with open(os.path.abspath(save), "w") as file:
                file.write(output_text)
                file.close()
        else:
            with open(os.path.abspath(save) + "/output.txt", "w") as file:
                file.write(output_text)
                file.close()

    return output_text
