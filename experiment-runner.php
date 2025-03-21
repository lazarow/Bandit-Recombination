<?php

$seeds = [4558567, 1226832, 4027462, 3754187, 3076827, 5393253, 3902955, 3396399, 2021843, 8772388, 3653387, 1052791, 7210496, 620540, 4760529, 4484595, 6241564, 682367, 1967568, 4213543, 3855553, 123078, 5502856, 624069, 1487545, 7123763, 8578532, 7654466, 3789507, 6821232];

$dataset_filepath = 'datasets/e401600.json';

for ($i = 0; $i < 30; $i++) {
    $command = "ga_gap.exe " . $dataset_filepath . " --seed " . $seeds[$i];
    $output = shell_exec($command);
    $lines = explode("\n", trim($output));
    $total_cost = "";
    $pbr_improvements = "";
    $total_improvements = "";
    foreach ($lines as $line) {
        if (strpos($line, "Total cost:") !== false) {
            $total_cost = strtr($line, ['Total cost: ' => '']);
        }
        if (strpos($line, "PBR improvements:") !== false) {
            $pbr_improvements = strtr($line, ['PBR improvements: ' => '']);
        }
        if (strpos($line, "Total improvements:") !== false) {
            $total_improvements = strtr($line, ['Total improvements: ' => '']);
        }
    }
    echo $total_cost . "\t" . $pbr_improvements . "\t" . $total_improvements . "\n";
}
echo '+++++++++++++++++++++' . "\n";

for ($i = 0; $i < 30; $i++) {
    $command = "ga_gap.exe " . $dataset_filepath . " --seed " . $seeds[$i] . " --pbr-offsprings-size 10";
    $output = shell_exec($command);
    $lines = explode("\n", trim($output));
    $total_cost = "";
    $pbr_improvements = "";
    $total_improvements = "";
    foreach ($lines as $line) {
        if (strpos($line, "Total cost:") !== false) {
            $total_cost = strtr($line, ['Total cost: ' => '']);
        }
        if (strpos($line, "PBR improvements:") !== false) {
            $pbr_improvements = strtr($line, ['PBR improvements: ' => '']);
        }
        if (strpos($line, "Total improvements:") !== false) {
            $total_improvements = strtr($line, ['Total improvements: ' => '']);
        }
    }
    echo $total_cost . "\t" . $pbr_improvements . "\t" . $total_improvements . "\n";
}
