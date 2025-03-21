<?php

$seeds = [4558567, 1226832, 4027462, 3754187, 3076827, 5393253, 3902955, 3396399, 2021843, 8772388, 3653387, 1052791, 7210496, 620540, 4760529, 4484595, 6241564, 682367, 1967568, 4213543, 3855553, 123078, 5502856, 624069, 1487545, 7123763, 8578532, 7654466, 3789507, 6821232];

$dataset_filepath = 'datasets/gamma_200_20_3.json';

for ($i = 0; $i < 30; $i++) {
    $command = "ga_gap.exe " . $dataset_filepath . " --seed " . $seeds[$i];
    $output = shell_exec($command);
    $lines = explode("\n", trim($output));
    $last_line = "";
    for ($j = count($lines) - 1; $j >= 0; $j--) {
        if (trim($lines[$j]) !== "") {
            $last_line = $lines[$j];
            break;
        }
    }
    echo strtr($last_line, ['Total cost: ' => '']) . "\n";
}
echo '+++++++++++++++++++++' . "\n";

for ($i = 0; $i < 30; $i++) {
    $command = "ga_gap.exe " . $dataset_filepath . " --seed " . $seeds[$i] . " --pbr-offsprings-size 10";
    $output = shell_exec($command);
    $lines = explode("\n", trim($output));
    $last_line = "";
    for ($j = count($lines) - 1; $j >= 0; $j--) {
        if (trim($lines[$j]) !== "") {
            $last_line = $lines[$j];
            break;
        }
    }
    echo strtr($last_line, ['Total cost: ' => '']) . "\n";
}
