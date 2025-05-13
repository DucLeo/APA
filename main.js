const w1 = [[-0.6115623,0.45754996,-0.0014264778,0.14003427,0.15973702,-0.019264808],
[-0.08952248,-0.7671072,0.37560645,-0.22811566,2.2616472,-0.028973263],
[-0.84124666,3.740101,0.3996625,-1.4696987,-2.4129918,-1.4735105],
[-0.32731274,-1.0239178,-0.7904756,0.03994351,1.5294149,0.8185676],
[-0.5872565,-0.62590617,-0.037713036,-0.36759424,0.0123318685,0.43887696],
[-0.36142075,0.33844396,-0.3502426,0.59112674,-0.09096527,-0.17723241]];

const b1 = [[-0.9630651],
[-0.45685032],
[-0.44839302],
[-0.48087108],
[1.6395983],
[-1.0532466]];

const w2 = [[-1.1493142,-0.06574549,-0.04000144,1.1372206],
[0.016742302,0.37116966,0.090217575,0.27274272],
[-0.60436165,0.32191405,0.3820969,1.2856364],
[0.43564516,0.28189597,-0.82985526,0.6345047],
[0.36218667,-0.71066743,0.2244135,-0.5986203],
[-0.57660013,0.44976327,-0.24838531,0.52474284]];

const b2 = [[0.60371196],
[-0.078738146],
[0.5372923],
[-0.6001593]];

const w3 = [[0.016813055,-0.2785336,-0.63024443],
[0.25016698,1.0654123,0.4800932],
[0.35065714,0.51379234,-0.23633045],
[-0.35780722,-0.58718586,0.25343198]];

const b3 = [[0.4695743],
[-0.004799305],
[-0.80262995]];

function selu(x){
    lamda = 1.0507009873554805;
    alpha = 1.6732632423543772;
    if (Array.isArray(x)){
        return x.map(x => x > 0 ? lamda * x : lamda * alpha * (Math.exp(x) - 1));
    }
    else {
        return x > 0 ? lamda * x : lamda * alpha * (Math.exp(x) - 1);
    }
}

function leaky_relu(x){
    alpha = 0.3;
    if (Array.isArray(x)){
        return x.map(x => x >= 0 ? x : alpha * x);
    }
    else {
        return x >= 0 ? x : alpha * x;
    }
}

function softmax(input){
    const maxInput = Math.max(...input);
    const exps = input.map(x => Math.exp(x - maxInput));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sumExps);
}

function argmax(output){
    let max = output[0];
    let maxIndex = 0;
    for (let i = 1; i < output.length; i++) {
        if (output[i] > max) {
            max = output[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

function summary(inputs, weights, biases){
    numInput = inputs.length;
    rowsWeights = weights.length;
    colsWeights = weights[0].length;
    numBiases = biases.length;
    if (numInput !== rowsWeights) {
        throw new Error("Weights quantity is not appropriate!");
    }
    if (colsWeights !== numBiases) {
        throw new Error("Quantities weights and biases do not match!");
    }
    const result = new Array(numBiases).fill(0);
    for (let i = 0; i < numBiases; i++) {
        for (let j = 0; j < numInput; j++) {
            result[i] += inputs[j] * weights[j][i];
        }
        result[i] += biases[i][0];
    }
    return result;
}

function analyzePA() {
    const age = parseFloat(document.getElementById('age').value);
    const rhr = parseFloat(document.getElementById('rhr').value);
    const hr = parseFloat(document.getElementById('hr').value);
    const vma = parseFloat(document.getElementById('vma').value);
    const steps = parseFloat(document.getElementById('steps').value);
    const iStanding = parseFloat(document.getElementById('iStanding').value);
    const iSitting = parseFloat(document.getElementById('iSitting').value);
    const iLying = parseFloat(document.getElementById('iLying').value);

    if (isNaN(age) || isNaN(rhr) || isNaN(hr) || isNaN(vma) || isNaN(steps) || isNaN(iStanding) || isNaN(iSitting) || isNaN(iLying)) {
        document.getElementById('result').textContent = "Недостаточно информации!";
        document.getElementById('overlay').style.display = 'flex';
        return;
    }

    const hrm = 208 - 0.7 * age;
    const hrr = (hr - rhr) / (hrm - rhr) * 100;

    inputs = [vma, iStanding, iSitting, iLying, steps, hrr];
    summary_layer1 = summary(inputs, w1, b1);
    output_layer1 = selu(summary_layer1);
    summary_layer2 = summary(output_layer1, w2, b2);
    output_layer2 = leaky_relu(summary_layer2);
    summary_layer3 = summary(output_layer2, w3, b3);
    output_layer3 = softmax(summary_layer3);
    final_type_active = argmax(output_layer3);

    document.getElementById('confirm1').textContent = "VMA:" + vma + " cm/s\u00B2";
    document.getElementById('confirm2').textContent = "iStanding:" + iStanding;
    document.getElementById('confirm3').textContent = "iSitting:" + iSitting;
    document.getElementById('confirm4').textContent = "iLying :" + iLying;
    document.getElementById('confirm5').textContent = "Steps:" + steps;
    document.getElementById('confirm6').textContent = "HRR:" + hrr.toFixed(2) + " %";
    document.getElementById('overlay').style.display = 'flex';
    
    if (argmax(output_layer3) == 0){

        document.getElementById('result').textContent = "НЕАКТИВНЫЙ\nСущественно не отличается от состояния покоя.\nВ течение дня на неделе требуются другие виды активности более высокого уровня.";
        document.getElementById('overlay').style.display = 'flex';
    }
    else if (argmax(output_layer3) == 1) {
        document.getElementById('result').textContent = "УМЕРЕННЫЙ\nСогласно рекомендациям ВОЗ, взрослым следует уделять этим видам активности 150–300 минут в неделю,\nчто эквивалентно 20–40 минутам в день, или больше, чтобы снизить риск заболеваний и улучшить здоровье.";
        document.getElementById('overlay').style.display = 'flex';
    }
    else {
        document.getElementById('result').textContent = "АКТИВНЫЙ\nСогласно рекомендациям ВОЗ, взрослым следует уделять этим видам активности 150–300 минут в неделю,\nчто эквивалентно 20–40 минутам в день, или больше, чтобы снизить риск заболеваний и улучшить здоровье.\nПоддержание такого уровня активности в течение дня в течение длительного времени поможет улучшить качество сна,\nбыстрее заснуть и увеличить продолжительность сна.";
        document.getElementById('overlay').style.display = 'flex';
    }
}
        
function quitResult() {
    document.getElementById('overlay').style.display = 'none';
}