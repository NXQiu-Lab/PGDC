#!/bin/bash
# slurm_queue_driver.sh — drive prepared CP2K single-point jobs through SLURM
# with bounded concurrency. Designed to run on the cluster head node inside
# tmux so the local laptop can disconnect at any time:
#
#   tmux new-session -d -s relabel \
#     "bash slurm_queue_driver.sh /path/to/base 16 60 relabel_sp_row_"
#
# Args:
#   $1 BASE_DIR       directory containing the prepared <GLOB>* job dirs
#   $2 MAX_CONCURRENT max number of jobs in the SLURM queue at once (def. 8)
#   $3 POLL_SECONDS   sleep between cycles (def. 60)
#   $4 GLOB           job-directory prefix glob (def. relabel_sp_row_)
#   $5 EXPECTED       total number of job dirs expected; the driver only exits
#                     when nothing is pending/submitted AND at least EXPECTED
#                     dirs exist (def. 0 = exit as soon as the queue empties).
#                     Set this when inputs are still being uploaded.
#
# Each job dir holds input.inp + submit.slurm and a `state` file with one of:
#   pending | submitted <jobid> | done | failed: <reason>
# A job whose SLURM job vanishes without a complete output is resubmitted,
# up to MAX_ATTEMPTS times. Progress is appended to driver.log and the latest
# summary is written to driver.status.

set -u

BASE="${1:?usage: slurm_queue_driver.sh BASE_DIR [MAX_CONCURRENT] [POLL_SECONDS] [GLOB] [EXPECTED]}"
LIMIT="${2:-8}"
POLL="${3:-60}"
GLOB="${4:-relabel_sp_row_}"
EXPECTED="${5:-0}"
MAX_ATTEMPTS=3
LOG="$BASE/driver.log"
STATUS="$BASE/driver.status"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

job_active() { # $1 = jobid
    [ -n "$(squeue -j "$1" -h 2>/dev/null)" ]
}

out_complete() { # $1 = job dir
    local f="$1/out"
    [ -f "$f" ] || return 1
    grep -q "SCF run converged" "$f" \
        && grep -q "STRESS|" "$f" \
        && grep -q "FORCES| Atomic forces" "$f"
}

log "driver start: BASE=$BASE LIMIT=$LIMIT POLL=$POLL GLOB=$GLOB"

while true; do
    # Pass 1: refresh states of submitted jobs.
    for d in "$BASE"/$GLOB*/; do
        [ -d "$d" ] || continue
        state="pending"
        [ -f "$d/state" ] && state=$(cat "$d/state")
        case "$state" in
            submitted*)
                jid=$(awk '{print $2}' "$d/state")
                if job_active "$jid"; then
                    :
                elif out_complete "$d"; then
                    echo "done" > "$d/state"
                    log "done: $d"
                else
                    att=0
                    [ -f "$d/attempts" ] && att=$(cat "$d/attempts")
                    if [ "$att" -lt "$MAX_ATTEMPTS" ]; then
                        echo "pending" > "$d/state"
                        log "job $jid vanished without complete output; will resubmit: $d (attempt $((att+1)))"
                    else
                        echo "failed: job vanished without complete output after $att attempts" > "$d/state"
                        log "FAILED: $d (job $jid vanished, attempts exhausted)"
                    fi
                fi
                ;;
        esac
    done

    # Pass 2: count active, submit pendings up to the limit.
    active=0
    for d in "$BASE"/$GLOB*/; do
        [ -d "$d" ] || continue
        state="pending"
        [ -f "$d/state" ] && state=$(cat "$d/state")
        case "$state" in
            submitted*)
                jid=$(awk '{print $2}' "$d/state")
                job_active "$jid" && active=$((active+1))
                ;;
        esac
    done

    for d in "$BASE"/$GLOB*/; do
        [ "$active" -lt "$LIMIT" ] || break
        [ -d "$d" ] || continue
        state="pending"
        [ -f "$d/state" ] && state=$(cat "$d/state")
        [ "$state" = "pending" ] || continue
        att=0
        [ -f "$d/attempts" ] && att=$(cat "$d/attempts")
        if [ "$att" -ge "$MAX_ATTEMPTS" ]; then
            echo "failed: sbatch attempts exhausted" > "$d/state"
            log "FAILED: $d (sbatch attempts exhausted)"
            continue
        fi
        echo $((att+1)) > "$d/attempts"
        jid=$(cd "$d" && sbatch --parsable submit.slurm 2>>"$LOG" | tail -1 | cut -d';' -f1)
        if [ -n "$jid" ]; then
            echo "submitted $jid" > "$d/state"
            active=$((active+1))
            log "submitted: $d job $jid (attempt $((att+1)))"
        else
            log "sbatch failed: $d (attempt $((att+1)))"
        fi
    done

    # Summarize.
    n_done=0; n_failed=0; n_sub=0; n_pend=0
    for d in "$BASE"/$GLOB*/; do
        [ -d "$d" ] || continue
        state="pending"
        [ -f "$d/state" ] && state=$(cat "$d/state")
        case "$state" in
            done)      n_done=$((n_done+1));;
            failed*)   n_failed=$((n_failed+1));;
            submitted*) n_sub=$((n_sub+1));;
            *)         n_pend=$((n_pend+1));;
        esac
    done
    n_total=$((n_done + n_failed + n_sub + n_pend))
    summary="done=$n_done failed=$n_failed submitted=$n_sub pending=$n_pend active=$active total=$n_total/$EXPECTED"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $summary" > "$STATUS"
    log "status: $summary"

    if [ "$n_pend" -eq 0 ] && [ "$n_sub" -eq 0 ] && [ "$n_total" -ge "$EXPECTED" ]; then
        log "ALL DONE: $n_done done, $n_failed failed"
        echo "$(date '+%Y-%m-%d %H:%M:%S') ALL DONE: $summary" >> "$STATUS"
        break
    fi
    sleep "$POLL"
done
