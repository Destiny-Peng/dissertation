"""Reservation budget, handoff, lifecycle and Run option tests; no GPU needed."""
from contextlib import nullcontext
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent/'lf3r_annotator'))
import gpu_memory_reservation as reservation
from baseline_jobs import BaselineJobsMixin
from backend_core import ValidationError


class FakeTorch:
    uint8 = 'uint8'
    def __init__(self, snapshots, fail=None):
        self.snapshots = snapshots
        self.device_index = 0
        self.allocations = []
        self.fail = fail
        self.cuda = SimpleNamespace(device_count=lambda: len(snapshots), device=self.device,
                                    init=lambda: None, mem_get_info=lambda: snapshots[self.device_index],
                                    synchronize=lambda: None, empty_cache=mock.Mock())
    def device(self, index):
        self.device_index = index
        return nullcontext()
    def empty(self, size, **kwargs):
        if self.device_index == self.fail:
            raise RuntimeError('simulated OOM')
        self.allocations.append((self.device_index, size))
        return object()


class ReservationTests(unittest.TestCase):
    def test_auto_budget_has_reserved_handoff_margin_and_same_tp_fraction(self):
        mib = reservation.MIB
        torch = FakeTorch([(12000*mib,20000*mib), (8000*mib,20000*mib)])
        allocations, budget = reservation.hold_memory(torch,.8,2)
        self.assertEqual(len(allocations),2)
        self.assertAlmostEqual(budget['resolved_total_fraction'],.1952)
        self.assertEqual(budget['reserved_mib_per_gpu'],[5952,5952])
        self.assertEqual(budget['vllm_target_mib_per_gpu'],[3904,3904])
        self.assertEqual(torch.allocations[0][1],5952*mib)

    def test_fixed_size_and_partial_oom_cleanup(self):
        mib = reservation.MIB
        torch = FakeTorch([(10000*mib,20000*mib)])
        _, budget = reservation.hold_memory(torch,.8,1,6000)
        self.assertEqual(budget['reserved_mib_per_gpu'],[6000])
        self.assertEqual(budget['vllm_target_mib_per_gpu'],[3952])
        with self.assertRaises(ValueError):
            reservation.hold_memory(torch,.8,1,9000)
        failing = FakeTorch([(10000*mib,20000*mib)]*2,fail=1)
        with self.assertRaisesRegex(RuntimeError,'OOM'):
            reservation.hold_memory(failing,.8,2)
        self.assertEqual(failing.cuda.empty_cache.call_count,2)

    def test_keeper_start_environment_and_close_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget={'resolved_total_fraction':.2}
            process = SimpleNamespace(stdout=io.StringIO(json.dumps({'event':'gpu_reservation_ready','budget':budget})+'\n'),
                                      stdin=io.StringIO(),wait=mock.Mock(return_value=0))
            keeper=reservation.MemoryReservation(Path(sys.executable),root,{'PYTHONPATH':'existing'},root/'run.log',.8,1)
            with mock.patch.object(reservation.subprocess,'Popen',return_value=process):
                self.assertEqual(keeper.start(),budget)
            env=keeper.worker_env()
            self.assertEqual(env['VLLM_WORKER_MULTIPROC_METHOD'],'spawn')
            self.assertIn('existing',env['PYTHONPATH'])
            self.assertEqual(env[reservation.SOCKET_ENV],str(keeper.path))
            keeper.close()
            self.assertTrue(process.stdin.closed)
            process.wait.assert_called_once()
            failed=SimpleNamespace(stdout=io.StringIO(''),stdin=io.StringIO(),wait=mock.Mock(return_value=1))
            keeper=reservation.MemoryReservation(Path(sys.executable),root,{},root/'run.log',.8,1)
            with mock.patch.object(reservation.subprocess,'Popen',return_value=failed):
                with self.assertRaisesRegex(RuntimeError,'reservation failed'):
                    keeper.start()
            self.assertTrue(failed.stdin.closed)
            failed=SimpleNamespace(stdout=io.StringIO(json.dumps({'event':'gpu_reservation_failed','error':'GPU 0 has insufficient free memory'})+'\n'),
                                   stdin=io.StringIO(),wait=mock.Mock(return_value=1))
            keeper=reservation.MemoryReservation(Path(sys.executable),root,{},root/'run.log',.8,1)
            with mock.patch.object(reservation.subprocess,'Popen',return_value=failed):
                with self.assertRaisesRegex(RuntimeError,'insufficient free memory'):
                    keeper.start()
            self.assertTrue(failed.stdin.closed)

    def test_cancel_cleanup_escalates_only_own_keeper(self):
        with tempfile.TemporaryDirectory() as directory:
            keeper=reservation.MemoryReservation(Path(sys.executable),Path(directory),{},Path(directory)/'run.log',.8,1)
            process=SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO(),terminate=mock.Mock(),kill=mock.Mock(),
                wait=mock.Mock(side_effect=[reservation.subprocess.TimeoutExpired('keeper',5),0]))
            keeper.process=process
            keeper.close()
            process.terminate.assert_called_once()
            process.kill.assert_not_called()

    def test_release_client_authenticates_and_fails_closed(self):
        stream=mock.Mock()
        stream.readline.return_value=b'{"released":true}\n'
        connection=mock.MagicMock()
        connection.__enter__.return_value=connection
        connection.makefile.return_value=stream
        with mock.patch.dict(reservation.os.environ,{reservation.SOCKET_ENV:'sock',reservation.TOKEN_ENV:'secret'}), \
             mock.patch.object(reservation.socket,'socket',return_value=connection):
            reservation.release_reserved_device(1)
            sent=json.loads(connection.sendall.call_args.args[0])
            self.assertEqual(sent,{'token':'secret','device':1})
            stream.readline.return_value=b'{"released":false}\n'
            with self.assertRaisesRegex(RuntimeError,'handoff failed'):
                reservation.release_reserved_device(1)

    def test_reserved_budget_reaches_vllm_without_requerying_occupied_memory(self):
        import robo_dopamine_persistent_worker as worker
        with tempfile.TemporaryDirectory() as directory:
            official = SimpleNamespace(LLM=mock.Mock(return_value="engine"))
            official.GRMInference = lambda path, **kwargs: official.LLM(model=path)
            args = SimpleNamespace(repo=Path(directory),model_path=Path(directory),tp=1,
                vllm_total_memory_fraction=.25,vllm_memory_safety_buffer_mib=2048,
                memory_budget={"scope":"reserved_gpu_memory"},extract_latent=False)
            with mock.patch.dict(sys.modules,{"examples":SimpleNamespace(inference=official), "examples.inference":official}), \
                 mock.patch.dict(reservation.os.environ,{reservation.SOCKET_ENV:"sock"}), \
                 mock.patch.object(worker,"resolve_worker_vllm_memory_budget") as requery:
                llm = official.LLM
                self.assertEqual(worker.initialize_robo_model(args),"engine")
                self.assertEqual(llm.call_args.kwargs["gpu_memory_utilization"],.25)
                self.assertEqual(llm.call_args.kwargs["worker_cls"],"gpu_reservation_worker.ReservedGPUWorker")
                requery.assert_not_called()

    def test_gpu_worker_releases_before_base_cuda_initialization(self):
        events=[]
        class Base:
            local_rank=1
            def init_device(self):
                events.append("cuda_init")
                return "ready"
        spec=importlib.util.spec_from_file_location("reservation_worker_test",TOOLS/"gpu_reservation_worker.py")
        module=importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules,{"vllm.v1.worker.gpu_worker":SimpleNamespace(Worker=Base)}):
            spec.loader.exec_module(module)
        with mock.patch.object(module,"release_reserved_device",side_effect=lambda device:events.append(f"release_{device}")):
            self.assertEqual(module.ReservedGPUWorker().init_device(),"ready")
        self.assertEqual(events,["release_1","cuda_init"])

    def test_runner_uses_reserved_budget_and_always_cleans_up(self):
        import robo_dopamine_runner as runner
        for failure in (None, "reserve", "worker"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                args=SimpleNamespace(gpu="2",vllm_free_memory_fraction=.8,dry_run=False,
                    robo_reserve_gpu_memory=True,robo_reserve_mib=6144,tensor_parallel_size=1)
                keeper=mock.Mock()
                keeper.start.return_value={"resolved_total_fraction":.2,"scope":"reserved_gpu_memory"}
                keeper.worker_env.return_value={"reservation":"active"}
                if failure == "reserve": keeper.start.side_effect=RuntimeError("cannot reserve")
                with mock.patch.object(reservation,"MemoryReservation",return_value=keeper), \
                     mock.patch.object(runner,"build_job_specs",return_value=[{"rollout_id":"r"}]), \
                     mock.patch.object(runner,"build_worker_command",return_value=["worker","--memory-budget-json","{}"]), \
                     mock.patch.object(runner,"run_streamed",side_effect=RuntimeError("worker failed") if failure=="worker" else None,return_value=0) as run, \
                     mock.patch.object(runner,"finalize_run",return_value=0) as finalize:
                    runner.run_persistent(args=args,config={"repo":root,"python":Path(sys.executable)},
                        records=[{}],model_path=root,run_root=root,raw_root=root,log_path=root/"log",
                        metadata_path=root/"run.json",jobs_path=root/"jobs.jsonl",commands_path=root/"commands.jsonl",
                        env={},metadata={})
                keeper.close.assert_called_once()
                expected=0 if failure is None else runner.ROBODOPAMINE_FATAL_EXIT_CODE
                self.assertEqual(finalize.call_args.kwargs["worker_return_code"],expected)
                if failure=="reserve": run.assert_not_called()
                else:
                    command=run.call_args.args[0]
                    self.assertEqual(command[command.index("--vllm-total-memory-fraction")+1],"0.2")
                    self.assertEqual(run.call_args.args[2],{"reservation":"active"})
                    plan=json.loads((root/"robo_dopamine_jobs.jsonl").read_text())
                    self.assertIn("--vllm-total-memory-fraction",plan["argv"])

    def test_run_options_reject_invalid_reservations(self):
        service=BaselineJobsMixin()
        options={'robo_reserve_gpu_memory':True,'robo_reserve_mib':12288}
        self.assertEqual(service._validate_options('robo_dopamine',options),options)
        for options in ({'robo_reserve_gpu_memory':'yes'}, {'robo_reserve_gpu_memory':True,'robo_reserve_mib':1024},
                        {'robo_reserve_mib':12288},{'robo_reserve_gpu_memory':True,'robo_reserve_mib':-1}):
            with self.assertRaises(ValidationError):
                service._validate_options('robo_dopamine',options)
        with self.assertRaises(ValidationError):
            service._validate_options('safe',{'robo_reserve_gpu_memory':True})


if __name__ == '__main__':
    unittest.main()
