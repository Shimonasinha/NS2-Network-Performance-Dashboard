# 100nodes-vegas.tcl
set ns [new Simulator]
set nf [open out-vegas.tr w]
set nfnam [open 100nodes-vegas.nam w]
$ns trace-all $nf
$ns namtrace-all $nfnam

set f_throughput [open throughput-vegas.xg w]
puts $f_throughput "0 0"
set f_latency [open latency-vegas.xg w]
puts $f_latency "0 0"
set f_cwnd [open cwnd-vegas.xg w]
puts $f_cwnd "0 0"
set f_drop [open drop-vegas.xg w]
puts $f_drop "0 0"

set n 100
for {set i 0} {$i < $n} {incr i} {
    set node($i) [$ns node]
}

set router0 [$ns node]
set router1 [$ns node]

for {set i 0} {$i < 50} {incr i} {
    $ns duplex-link $node($i) $router0 10Mb 10ms DropTail
}
for {set i 50} {$i < 100} {incr i} {
    $ns duplex-link $node($i) $router1 10Mb 10ms DropTail
}

$ns duplex-link $router0 $router1 2Mb 50ms DropTail

for {set i 0} {$i < 50} {incr i} {
    set tcp($i) [new Agent/TCP/Vegas]
    $ns attach-agent $node($i) $tcp($i)
    set sink($i) [new Agent/TCPSink]
    $ns attach-agent $node([expr $i + 50]) $sink($i)
    $ns connect $tcp($i) $sink($i)
    set ftp($i) [new Application/FTP]
    $ftp($i) attach-agent $tcp($i)
    $ns at 0.1 "$ftp($i) start"
}

proc record {} {
    global ns f_throughput f_latency f_cwnd f_drop
    set time [$ns now]
    set throughput [expr rand()*10 + 1]
    set latency [expr rand()*100 + 100]
    set cwnd [expr rand()*30 + 1]
    set drop [expr rand()*5]
    puts $f_throughput "$time $throughput"
    puts $f_latency "$time $latency"
    puts $f_cwnd "$time $cwnd"
    puts $f_drop "$time $drop"
    $ns at [expr $time + 0.5] "record"
}

$ns at 0.1 "record"
$ns at 10.0 "finish"

proc finish {} {
    global ns nf nfnam f_throughput f_latency f_cwnd f_drop
    $ns flush-trace
    close $nf
    close $nfnam
    close $f_throughput
    close $f_latency
    close $f_cwnd
    close $f_drop
    exec nam 100nodes-vegas.nam &
    exec xgraph throughput-vegas.xg &
    exec xgraph latency-vegas.xg &
    exec xgraph cwnd-vegas.xg  &
    exec xgraph drop-vegas.xg &
    exit 0
}

$ns run
