# (C) Copyright 2020- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Some sanity tests for :any:`JobBuilder`/:any:`job_options`
"""

import click
from click.testing import CliRunner
import pytest
import yaml

from ifsbench.command_line.click_job_options import JobBuilder, job_options

from ifsbench import CpuBinding, CpuDistribution, Job


def cli_test(tmp_path, cmd_flags):
    """
    Helper function for passing the given flags via commandline and
    retrieving the resulting JobBuilder object.
    """
    yaml_path = tmp_path / 'result.yaml'

    @click.command('click_test')
    @job_options
    def test_command(job_builder):
        with yaml_path.open('w', encoding='utf-8') as f:
            yaml.dump(job_builder.dump_config(), stream=f, encoding='utf-8')

    runner = CliRunner()
    result = runner.invoke(test_command, cmd_flags, standalone_mode=False)

    if hasattr(result, 'output_bytes'):
        print(result.output_bytes)
    else:
        print(result.stdout_bytes, result.stderr_bytes)

    if result.exception:
        raise result.exception

    assert result.exit_code == 0

    with yaml_path.open('r', encoding='utf-8') as f:
        builder = JobBuilder.from_config(yaml.safe_load(f))

    return builder


def test_job_builder_no_flags():
    """
    Test the JobBuilder without any arguments.
    """
    builder = JobBuilder()

    assert builder.build_job() == Job()


@pytest.mark.parametrize(
    'flags_in,ref_job',
    [
        (
            ['--tasks', '4', '--nodes', '1'],
            Job(tasks=4, nodes=1),
        ),
        (
            ['--tasks-per-node', '8', '--account', 'myaccount', '--partition', 'mypartition'],
            Job(tasks_per_node=8, account='myaccount', partition='mypartition'),
        ),
        (
            ['--cpus-per-task', '2', '--threads-per-core', '2', '--gpus-per-node', '4'],
            Job(cpus_per_task=2, threads_per_core=2, gpus_per_node=4),
        ),
        (
            ['--bind', 'cores', '--distribute-remote', 'block', '--distribute-local', 'cyclic'],
            Job(
                bind=CpuBinding.BIND_CORES,
                distribute_remote=CpuDistribution.DISTRIBUTE_BLOCK,
                distribute_local=CpuDistribution.DISTRIBUTE_CYCLIC,
            ),
        ),
    ],
)
def test_job_builder_from_flags(tmp_path, flags_in, ref_job):
    """
    Test the job_options/JobBuilder pipeline by passing given flags and
    comparing the resulting Job object against an explicit reference.
    """
    builder = cli_test(tmp_path, flags_in)

    assert builder.build_job() == ref_job

@pytest.mark.parametrize(
    'flags_in',
    [
        ('--tasks', 'many'),
        ('--bind', 'wall'),
        ('--bind', 4),
        ('--cpus_per_task', 0.2),
        ('--distribute_remote', '5')
    ]
)
def test_job_builder_invalid_values(tmp_path, flags_in):
    """
    Test that job_option fails if invalid options are passed.
    """
    with pytest.raises(Exception):
        cli_test(tmp_path, flags_in)


@pytest.mark.parametrize(
    'flags_in,default_job,ref_job',
    [
        (
            [],
            Job(tasks=4, nodes=1, account='default_account'),
            Job(tasks=4, nodes=1, account='default_account'),
        ),
        (
            ['--tasks', '8'],
            Job(tasks=4, nodes=1, account='default_account'),
            Job(tasks=8, nodes=1, account='default_account'),
        ),
        (
            ['--account', 'override_account', '--partition', 'override_partition'],
            Job(tasks=4, nodes=1, account='default_account'),
            Job(tasks=4, nodes=1, account='override_account', partition='override_partition'),
        ),
    ],
)
def test_job_builder_build_job_overrides_default(tmp_path, flags_in, default_job, ref_job):
    """
    Test the job_options/JobBuilder pipeline by passing given flags and an
    initial Job object and comparing the resulting Job object against an
    explicit reference.
    """
    builder = cli_test(tmp_path, flags_in)

    job = builder.build_job(default_job=default_job)

    assert job == ref_job
